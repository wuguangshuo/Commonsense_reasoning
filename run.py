import torch
from torch.utils.data import DataLoader
from transformers.optimization import AdamW,get_cosine_schedule_with_warmup,get_linear_schedule_with_warmup
from transformers import AutoTokenizer

import time
import random
from sklearn import metrics
import json
from tqdm import tqdm
from sklearn.model_selection import KFold,StratifiedKFold
import logging

from config import set_args
from utils import seed_everything,set_logger
from data_utils import load_dataset,CCKSDataset
from model import Model

args = set_args()
seed_everything(args.seed)
args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
set_logger(args.log_dir)
start_time = time.time()

#加载数据集 采取正样本不扩增而是直接过采样
train_path = args.data_dir + '/new.jsonl'  # 训练集
test_path = args.data_dir + '/dev_triple.jsonl'  # 测试集

train_data_all = load_dataset(train_path)
random.shuffle(train_data_all)
test_raw_data = load_dataset(test_path)

kf = KFold(n_splits=args.n_split)
kf_data = kf.split(train_data_all)

class FGM():
    def __init__(self, model):
        self.model = model
        self.backup = {}

    def attack(self, epsilon=0.3, emb_name='word_embeddings.'):
        # emb_name这个参数要换成你模型中embedding的参数名
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                self.backup[name] = param.data.clone()
                norm = torch.norm(param.grad)
                if norm != 0 and not torch.isnan(norm):
                    r_at = epsilon * param.grad / norm
                    param.data.add_(r_at)

    def restore(self, emb_name='word_embeddings.'):
        # emb_name这个参数要换成你模型中embedding的参数名
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}

# skf = StratifiedKFold(n_splits=args.n_split, shuffle=True, random_state=args.seed)
y=[]
for i in range(len(train_data_all)):
    y.append(train_data_all[i][2])
#构建模型
tokenizer=AutoTokenizer.from_pretrained(args.model_dir)
model = Model(args).to(args.device)
total_params = sum(p.numel() for p in model.parameters())
print(f'{total_params:,} total parameters.')
param_optimizer = list(model.named_parameters())
no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
optimizer_grouped_parameters = [
    {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.001},
    {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
]
optimizer = AdamW(optimizer_grouped_parameters, lr=args.lr)
if args.use_fgm:
    fgm = FGM(model)
#验证函数
def evaluate(args, model, data_iter):
    model.eval()
    loss_total = 0
    predicts, sents, grounds, all_bires = [], [], [], []
    with torch.no_grad():
        for i, batches in enumerate(tqdm(data_iter)):
            input_ids, attention_mask, type_ids, sent, _, labels = batches
            input_ids, attention_mask, type_ids, labels = \
                input_ids.to(args.device), attention_mask.to(args.device), type_ids.to(args.device), labels.to(
                    args.device)
            logits,loss = model(input_ids, attention_mask, type_ids,labels)
            loss_total += loss.item()
            _, bires = torch.max(logits, dim=-1)
            bires=bires.tolist()
            labels =labels.tolist()
            for b, g in zip(bires, labels):
                all_bires.append(b)
                grounds.append(g)
    print("test set size:", len(grounds))
    accuracy = metrics.accuracy_score(grounds, all_bires)
    p = metrics.precision_score(grounds, all_bires)
    r = metrics.recall_score(grounds, all_bires)
    f1 = metrics.f1_score(grounds, all_bires)
    print("f1:{},p:{},r,{}, accuracy:{}".format(f1, p, r, accuracy))
    logging.info("f1:{},p:{},r,{}, accuracy:{}".format(f1, p, r, accuracy))
    return f1,p,r, loss_total / len(data_iter)

kf_index=0
# for train_raw_dataindex, dev_raw_dataindex in kf_data:
for train_raw_dataindex, dev_raw_dataindex in kf_data:
    kf_index+=1
    train_raw_data=[]
    for index in train_raw_dataindex.tolist():
        train_raw_data.append(train_data_all[index])
    dev_raw_data = []
    for index in dev_raw_dataindex.tolist():
        dev_raw_data.append(train_data_all[index])

    train_data=CCKSDataset(train_raw_data,args,tokenizer)
    dev_data=CCKSDataset(dev_raw_data,args,tokenizer)
    test_data=CCKSDataset(test_raw_data,args,tokenizer)
    train_iter = DataLoader(train_data,shuffle=True,batch_size=args.batch_size,collate_fn=train_data.collate_fn)
    dev_iter = DataLoader(dev_data,shuffle=False,batch_size=args.batch_size,collate_fn=dev_data.collate_fn)
    test_iter = DataLoader(test_data,shuffle=False,batch_size=args.batch_size,collate_fn=test_data.collate_fn)
    train_steps_per_epoch = len(train_iter)
    # scheduler = get_cosine_schedule_with_warmup(optimizer,
    #                                             num_warmup_steps=(args.epochs // 10) * train_steps_per_epoch,
    #                                             num_training_steps=args.epochs * train_steps_per_epoch)
    scheduler = get_linear_schedule_with_warmup(optimizer,0,num_training_steps=args.epochs * train_steps_per_epoch)
    #模型训练
    total_batch=0
    dev_best_loss=1e12
    best_score=0.0
    print('kf [{}/{}]'.format(kf_index, 5))
    logging.info('kf [{}/{}]'.format(kf_index, 5))
    for epoch in range(args.epochs):
        print('Epoch [{}/{}]'.format(epoch + 1, args.epochs))
        logging.info('Epoch [{}/{}]'.format(epoch + 1, args.epochs))
        for i, batches in enumerate(tqdm(train_iter)):
            input_ids, attention_mask, type_ids,sent,_, labels = batches

            input_ids, attention_mask, type_ids, labels = \
                input_ids.to(args.device), attention_mask.to(args.device), type_ids.to(args.device), labels.to(args.device)

            if not args.use_fgm:
                logits,loss = model(input_ids, attention_mask, type_ids,labels)
                loss.backward()  # 反向传播，得到正常的grad
                optimizer.step()
                scheduler.step()
                model.zero_grad()
            if args.use_fgm:
                logits, loss = model(input_ids, attention_mask, type_ids,labels)
                loss.backward()  # 反向传播，得到正常的grad
                # 对抗训练
                fgm.attack()  # 在embedding上添加对抗扰动
                attention_mask=attention_mask.squeeze(dim=-1)
                logits_adv, loss_adv = model(input_ids, attention_mask, type_ids,labels)
                loss_adv.backward()  # 反向传播，并在正常的grad基础上，累加对抗训练的梯度
                fgm.restore()  # 恢复embedding参数
                # 梯度下降，更新参数
                optimizer.step()
                scheduler.step()
                model.zero_grad()

            total_batch += 1
            if total_batch % args.test_batch == 0:
                print("test:")
                f1,p,r,dev_loss= evaluate(args, model, dev_iter)
                score=f1
                print("loss", total_batch, loss.item(), dev_loss)
                if score > best_score:
                    print("save", score)
                    torch.save(model.state_dict(), args.output_dir + "model.ckpt")
                    best_score= score
                model.train()

#测试集
def predict(args, model, data_iter):
    model.eval()
    predicts = []
    with torch.no_grad():
        for i, batches in enumerate(tqdm(data_iter)):
            input_ids, attention_mask, type_ids, sent, id, _ = batches
            input_ids, attention_mask, type_ids=input_ids.to(args.device), attention_mask.to(args.device), type_ids.to(args.device)
            logits = model(input_ids, attention_mask, type_ids,labels=None)
            _, bires = torch.max(logits, dim=-1)
            for b, t in zip(bires, id):
                predicts.append({"triple_id": t,"salience": b.item()})
                # predicts.append({"salience": b.item(), "triple_id": t})
    with open(args.output_dir + "xx_result.jsonl", "w",encoding='utf-8') as f:
        for t in predicts:
            f.write(json.dumps(t, ensure_ascii=False)+"\n")


def eval(args, model, test_iter):
    # test
    model.load_state_dict(torch.load(args.output_dir + "model.ckpt"))
    predict(args, model, test_iter)

eval(args, model, test_iter)
















