from dataclasses import dataclass
from typing import Any, Callable, Dict, List, NewType, Tuple
import torch
from torch.nn.utils.rnn import pad_sequence
from transformers import PreTrainedTokenizer
import logging
import os
import pickle
import time
import torch
from filelock import FileLock
from torch.utils.data.dataset import Dataset
import json
import re
import random
import itertools
import numpy as np
import pickle

InputDataClass = NewType("InputDataClass", Any)
"""
A DataCollator is a function that takes a list of samples from a Dataset
and collate them into a batch, as a dictionary of Tensors.
"""
DataCollator = NewType("DataCollator", Callable[[List[InputDataClass]], Dict[str, torch.Tensor]])
SIMI_PinYin = pickle.load(open('simi_dis_pinyin.pkl', 'rb'))


@dataclass
class DataCollatorForMLM:
    """
    Data collator used for language modeling.
    - collates batches of tensors, honoring their tokenizer's pad_token
    - preprocesses batches for masked language modeling
    """

    tokenizer: PreTrainedTokenizer
    mlm: bool = True
    mlm_probability: float = 0.2

    def __call__(self, examples: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        in_puts = []
        lab_puts = []
        for example in examples:
            in_puts.append(example["input_ids"])
            lab_puts.append(example["labels"])
        in_batch = self._tensorize_batch(in_puts)
        lab_batch = self._tensorize_batch(lab_puts)
        inputs, labels, true_inputs = self.mask_tokens(in_batch, lab_batch)
        return {"input_ids": inputs, "labels": labels, "true_inputs": true_inputs}

    def _tensorize_batch(self, examples: List[torch.Tensor]) -> torch.Tensor:
        return pad_sequence(examples, batch_first=True, padding_value=self.tokenizer.pad_token_id)

    def mask_tokens(self, in_batch: torch.Tensor, lab_batch: torch.tensor) -> Tuple[
        torch.Tensor, torch.Tensor, torch.Tensor]:
        true_inputs = in_batch.clone()

        probability_matrix = torch.full(lab_batch.shape, self.mlm_probability)
        special_tokens_mask = [
            self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in lab_batch.tolist()
        ]
        probability_matrix.masked_fill_(torch.tensor(special_tokens_mask, dtype=torch.bool), value=0.0)
        if self.tokenizer._pad_token is not None:
            padding_mask = lab_batch.eq(self.tokenizer.pad_token_id)
            lab_batch[padding_mask] = -100
            probability_matrix.masked_fill_(padding_mask, value=0.0)
        masked_indices = torch.bernoulli(probability_matrix).bool()
        true_inputs[~masked_indices] = -100

        indices_replaced = torch.bernoulli(torch.full(lab_batch.shape, 0.5)).bool() & masked_indices
        in_batch[indices_replaced] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)

        indices_sim = torch.bernoulli(torch.full(lab_batch.shape, 0.6)).bool() & masked_indices & ~indices_replaced
        in_batch[indices_sim] = self.replace_sim1(in_batch[indices_sim])

        indices_random = masked_indices & ~indices_replaced & ~indices_sim
        in_batch[indices_random] = self.replace_sim2(in_batch[indices_random])

        return in_batch, lab_batch, true_inputs

    def replace_sim1(self, pinyin_list: torch.tensor) -> torch.tensor:
        simi_s = [['n', 'l', 'r'], ['j', 'q', 'x'], ['zh', 'z', 'ch', 'c', 'sh', 's'], ['b', 'p'], ['d', 't'],
                  ['m', 'f'],
                  ['g', 'k', 'h']]
        simi_y = [['ian', 'ie', 'iang'], ['ang', 'an'], ['ai', 'a'], ['eng', 'en'], ['ei', 'e'], ['ing', 'in'],
                  ['iu', 'ou'], ['iong', 'ong']]
        string_s = '('
        string_s += "".join([i + '|' for item in simi_s for i in item])
        string_s = string_s[:-1] + ')'
        string_y = '('
        string_y += "".join([i + '|' for item in simi_y for i in item])
        string_y = string_y[:-1] + ')\d$'
        res_list = []
        for in_pinyin in self.tokenizer.convert_ids_to_tokens(pinyin_list):

            sch_s = re.search(string_s, in_pinyin)
            out_pinyin = in_pinyin
            out_pinyin_s = in_pinyin
            out_pinyin_y = in_pinyin
            if sch_s:
                ori_s = sch_s.group(0)
                rep_s = ori_s
                for item in simi_s:
                    if ori_s in item:
                        t_item = item.copy()
                        t_item.remove(ori_s)
                        rep_s = random.choice(t_item)
                        break
                out_pinyin_s = re.sub(ori_s, rep_s, out_pinyin, count=1)
            if random.random() < 0.6:
                out_pinyin = out_pinyin_s
            sch_y = re.search(string_y, out_pinyin)
            if sch_y:
                ori_y = sch_y.group(0)[:-1]
                rep_y = ori_y
                for item in simi_y:
                    if ori_y in item:
                        t_item = item.copy()
                        t_item.remove(ori_y)
                        rep_y = random.choice(t_item)
                        break
                out_pinyin_y = re.sub(ori_y, rep_y, out_pinyin)
            if random.random() < 0.6:
                out_pinyin = out_pinyin_y
            if random.random() < 0.5:
                tune = str(random.randint(1, 5))
                out_pinyin = re.sub('\d', tune, out_pinyin)
            if out_pinyin in self.tokenizer.vocab:
                res_list.append(self.tokenizer.convert_tokens_to_ids(out_pinyin))
            else:
                res_list.append(self.tokenizer.convert_tokens_to_ids(in_pinyin))
        return torch.tensor(res_list, dtype=torch.long)


    def replace_sim2(self, pinyin_list: torch.tensor) -> torch.tensor:
        res_list = []
        for in_pinyin in self.tokenizer.convert_ids_to_tokens(pinyin_list):
            if in_pinyin[:-1] not in SIMI_PinYin.keys():
                # if consonant is None or vowel is None:
                res_list.append(self.tokenizer.convert_tokens_to_ids(in_pinyin))
                continue
            candidate = SIMI_PinYin[in_pinyin[:-1]]
            out_pinyin = random.choice(candidate)
            tone = int(in_pinyin[-1])
            # tone_list = [tone - 1, tone, tone + 1]
            if tone == 1 or tone == 5:
                tone_list = [1, 2, 5]
            elif tone == 2:
                tone_list = [1, 2, 3, 5]
            elif tone == 3:
                tone_list = [1, 2, 3, 4, 5]
            else:
                tone_list = [2, 3, 4]

            out_pinyin += str(random.choice(tone_list))
            if out_pinyin in self.tokenizer.vocab:
                res_list.append(self.tokenizer.convert_tokens_to_ids(out_pinyin))
            else:
                res_list.append(self.tokenizer.convert_tokens_to_ids(in_pinyin))
        return torch.tensor(res_list, dtype=torch.long)


class MLMDataset(Dataset):
    """
    This will be superseded by a framework-agnostic approach
    soon.
    """

    def __init__(self, con_tokenizer: PreTrainedTokenizer, lab_tokenizer: PreTrainedTokenizer, file_path: str):
        self.examples = []
        with open(file_path, encoding="utf-8") as f:
            line = f.readline()
            i = 0
            while line:
                # if i > 25000000:
                text = json.loads(line.strip())
                con = con_tokenizer.convert_tokens_to_ids(tokens=text[0])
                lab = lab_tokenizer.convert_tokens_to_ids(tokens=text[1])
                if len(lab) > 128:
                    line = f.readline()
                    continue
                self.examples.append({"input_ids": torch.tensor(con, dtype=torch.long),
                                      "labels": torch.tensor(lab, dtype=torch.long)})
                i += 1
                if i % 100000 == 0:
                    print(i)
                line = f.readline()

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return self.examples[i]
