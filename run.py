# -*- coding: utf-8 -*-
from transformers import BertConfig, BertTokenizer, AutoModelForMaskedLM, TrainingArguments
from utils import DataCollatorForMLM, MLMDataset
from train import Trainer
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
# from train_args import TrainingArguments
from middle_train import Middle_Trainer
import numpy
import torch
import torch.nn as nn
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--local_rank", type=int)
args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = '1,2,3'
config = BertConfig()
config.output_hidden_states = True
config.vocab_size = 41460
model = AutoModelForMaskedLM.from_config(config)
model.bert.embeddings.word_embeddings = nn.Embedding(1839, 768, padding_idx=0)
con_tokenizer = BertTokenizer.from_pretrained('y2d1')
lab_tokenizer = BertTokenizer.from_pretrained('z2d')
data_collator = DataCollatorForMLM(tokenizer=con_tokenizer, mlm=True, mlm_probability=0.2)
train_dataset = MLMDataset(con_tokenizer=con_tokenizer, lab_tokenizer=lab_tokenizer,
                           file_path='trainpath')
eval_dataset = MLMDataset(con_tokenizer=con_tokenizer, lab_tokenizer=lab_tokenizer,
                          file_path='evalpath')
training_args = TrainingArguments(
    output_dir='../results',  # output directory
    do_train=True,
    seed=42,
    num_train_epochs=3,  # total # of training epochs
    per_device_train_batch_size=32,  # batch size per device during training
    per_device_eval_batch_size=16,  # batch size for evaluation
    warmup_steps=500,  # number of warmup steps for learning rate scheduler
    weight_decay=0.01,  # strength of weight decay
    save_steps=10000,
    eval_steps=10000,
    evaluate_during_training=True,
    do_eval=True,
    dataloader_drop_last=True,
    # local_rank=args.local_rank
    local_rank=-1
)


def compute_metrics(pred):
    labels = pred.label_ids
    ind = numpy.where(labels != -100)
    preds = pred.predictions
    print(labels[ind], preds[ind])
    acc = accuracy_score(labels[ind], preds[ind])
    return {
        'accuracy': acc,
        'sample_len': len(ind[0])
    }


trainer = Middle_Trainer(model=model, args=training_args, data_collator=data_collator, train_dataset=train_dataset,
                         eval_dataset=eval_dataset, prediction_loss_only=False, compute_metrics=compute_metrics)
sd = trainer.train()
s = trainer.evaluate(eval_dataset)
print(s)
