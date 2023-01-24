from transformers import BertModel
import torch
import os
import requests


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")




pretrained = BertModel.from_pretrained("src/model/bert_base_Chinese").to(device)

for param in pretrained.parameters():
    param.requires_grad_(False)


# 定义下游任务模型
class MBertModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Linear(768, 16)

    def forward(self, input_ids, attention_mask, token_type_ids):
        with torch.no_grad():
            out = pretrained(input_ids, attention_mask, token_type_ids)
        out = self.fc(out.last_hidden_state[:, 0])
        out = out.softmax(dim=1)
        return out


if __name__ == '__main__':
    bertmodel = MBertModel()
