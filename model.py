import torch.nn as nn
from transformers import AutoModel
import torch


class LabelSmoothingLoss(nn.Module):
    def __init__(self, classes, smoothing=0.05, dim=-1):
        super(LabelSmoothingLoss, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.cls = classes
        self.dim = dim

    def forward(self, pred, target):
        pred = pred.log_softmax(dim=self.dim)
        with torch.no_grad():
            # true_dist = pred.data.clone()
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (self.cls - 1))
            true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        tmp=torch.sum(-true_dist * pred, dim=self.dim)
        return torch.mean(tmp)

class Model(nn.Module):
    def __init__(self, args):
        super(Model, self).__init__()
        self.num_labels = 1
        self.bert = AutoModel.from_pretrained(args.model_dir)
        self.dropout = nn.Dropout(args.dropout)
        self.dense_1 = nn.Linear(args.hidden_size, 2)
        self.label_smooth_loss = LabelSmoothingLoss(classes=2, smoothing=args.smoothing)

    def forward(self, input_ids, attention_mask, type_ids,labels):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=type_ids)
        # procuct_ouput=outputs.last_hidden_state[:, 0]
        sequence_output = outputs[0]
        # sequence_output = self.dropout(sequence_output)
        # procuct_ouput = torch.mean(sequence_output, 1)
        mask_2 = attention_mask  # 其余等于 1 的部分，即有效的部分
        mask_2_expand = mask_2.unsqueeze_(-1).expand(sequence_output.size()).float()
        sum_mask = mask_2_expand.sum(dim=1)  # 有效的部分“长度”求和
        sum_mask = torch.clamp(sum_mask, min=1e-9)
        bert_enc = torch.sum(sequence_output * mask_2_expand, dim=1) / sum_mask
        output = self.dropout(bert_enc)
        logits = self.dense_1(output)
        if labels is not  None:
            # loss = self.label_smooth_loss(logits, labels.view(-1))
            loss_fct = nn.CrossEntropyLoss()
            loss=loss_fct(logits, labels)
            return logits,loss
        else:
            return logits
