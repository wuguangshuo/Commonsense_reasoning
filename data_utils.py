import torch
from torch.utils.data import Dataset

import numpy as np

from tqdm import tqdm
import json

PAD, CLS, SEP = '[PAD]', '[CLS]', '[SEP]'
def load_dataset(path):
    contents = []
    with open(path, 'r', encoding='UTF-8') as f:
        for line in tqdm(f):
            lin = line.strip()
            if not lin:
                continue
            line_dict = json.loads(lin)
            subject = line_dict["subject"]
            object = line_dict["object"]
            predicate = line_dict["predicate"]
            triple_id = line_dict["triple_id"]
            raw_sent = SEP.join([subject, predicate, object])
            if "salience" in line_dict.keys():
                salience = line_dict["salience"]
                contents.append([raw_sent, triple_id, int(salience)])
                if int(salience)==1:
                    contents.append([raw_sent, triple_id, int(salience)])
            else:
                contents.append([raw_sent, triple_id, 0])
    return contents

class CCKSDataset(Dataset):
    def __init__(self,data,args,tokenizer):
        self.tokenizer = tokenizer
        self.device=args.device
        self.data=data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        feature = self.data[index]
        return feature

    def collate_fn(self,feature):
        sent=[x[0] for x in feature]
        id = [x[1] for x in feature]
        label=[x[2] for x in feature]
        encode_result = self.tokenizer(sent)

        batch_size=encode_result['input_ids']
        batch_len = len(batch_size)
        max_len = max([len(s) for s in encode_result['input_ids']])

        batch_data = 0 * np.ones((batch_len, max_len))
        batch_token_type = 0 * np.ones((batch_len, max_len))

        for j in range(batch_len):
            cur_len = len(encode_result['input_ids'][j])
            batch_data[j][:cur_len] = encode_result['input_ids'][j]

        for j in range(batch_len):
            cur_len = len(encode_result['input_ids'][j])
            batch_token_type[j][cur_len:] = 1


        input_ids = torch.tensor(batch_data, dtype=torch.long)
        attention_mask = torch.ne(input_ids, 0)
        type_ids = torch.tensor(batch_token_type, dtype=torch.long)
        label = torch.tensor(label, dtype=torch.long)
        return input_ids, attention_mask, type_ids,sent,id,label