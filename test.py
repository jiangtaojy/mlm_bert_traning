from transformers import BertConfig, TrainingArguments, BertModel, Trainer, AutoModelForMaskedLM, BertTokenizer
import torch.nn as nn
import torch
import torch.nn.functional as F
import heapq
import numpy
from pypinyin import pinyin, Style

config = BertConfig()
config.vocab_size = 41460  # 句子词典
model = AutoModelForMaskedLM.from_config(config)
model.bert.embeddings.word_embeddings = nn.Embedding(1839, 768, padding_idx=0)
state_dict = torch.load('./results/checkpoint-00000/pytorch_model.bin', map_location="cpu")
model.load_state_dict(state_dict)
model.eval()
pinyin_list = [i for tmp in pinyin('手机没电了', style=Style.TONE3, neutral_tone_with_five=True) for i in tmp]
con_tokenizer = BertTokenizer.from_pretrained('y2d1')
lab_tokenizer = BertTokenizer.from_pretrained('z2d')
con = torch.tensor(con_tokenizer.convert_tokens_to_ids(pinyin_list)).unsqueeze(0)
out_top5 = torch.topk(F.softmax(model(con)[0].squeeze(0), dim=-1), k=10)
values = out_top5[0].detach().numpy().tolist()
indices = out_top5[1].detach().numpy().tolist()
for i, item in enumerate(indices):
    print(lab_tokenizer.convert_ids_to_tokens(item))
    print(values[i])


