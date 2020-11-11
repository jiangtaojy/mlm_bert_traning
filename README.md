# mlm_bert_traning
基于mlm方式的带有纠错功能的拼音转汉字bert预训练模型

## 依赖

python==3.6.10

torch==1.4.0

tranformers==3.1.0

## 目的

将可能包含有错误的拼音解码成正确的汉字序列，可用于asr（语音识别）的拼音输出进行纠错。

## 训练

运行run.py，将其中的训练数据路径和测试数据路径改为你们的文件路径，文件格式类似trainpath和evalpath文件格式保持一致

测试

运行test.py，将拼音序列输入，输出为每个位置的前5个最可能token以及对应的概率
