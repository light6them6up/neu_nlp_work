import random
import warnings
import pandas as pd
import re
import numpy as np
import torch
import torch.nn as nn
import time

from sklearn.model_selection import train_test_split
from transformers import AdamW, get_linear_schedule_with_warmup
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from bs4 import BeautifulSoup
from torchtext.legacy import data
from torchtext.legacy import datasets
from transformers import BertTokenizer, BertModel

#读取数据集
warnings.filterwarnings('ignore')
data = pd.read_csv('./Desktop./zkm./labeledTrainData.tsv',header=0, delimiter="\t", quoting=3)
# print(type(data))

def dataSet_preprocess(review):
    #任务一：去掉html标记。
    raw_text = BeautifulSoup(review,'html').get_text()
    #任务二：去掉非字母字符,sub(pattern, replacement, string) 用空格代替
    letters = re.sub('[^a-zA-Z]',' ',raw_text)
    #str.split(str="", num=string.count(str)) 通过指定分隔符对字符串进行切片，如果参数 num 有指定值，则仅分隔 num 个子字符串
    #这里是先将句子转成小写字母表示，再按照空格划分为单词list
    words = letters.lower().split()
#     #获取停用词，从数据集中去掉停用词
#     stop_words_file = './Desktop./zkm./english.txt'
#     stopwords = get_stopwords(stop_words_file)
#     words = [word for word in words if word not in stopwords]
#     ' '.join(words)
    return words
#从停用词文件中获取停用词表
def get_stopwords(stop_words_file):
    with open(stop_words_file,encoding='utf-8') as f:
        stopwords = f.read()
    stopwords_list = stopwords.split('\n')
    custom_stopwords_list = [i for i in stopwords_list]
    return custom_stopwords_list

review_data = []
sentiment_data = []
#review_data存放评论
for review in data['review']:
    review_data.append(' '.join(dataSet_preprocess(review)))
#sentiment_data存放每条评论相对应的情感倾向：1为积极的，0为消极的
for sentiment in data['sentiment']:
    sentiment_data.append(sentiment)
# print(type(review_data))  #list

data["review"] = pd.DataFrame(review_data)
data["sentiment"] = pd.DataFrame(sentiment_data)
print(data)
print(type(data))

#截断数据操作
data["review"] = data["review"].str[:2000]
# data = pd.DataFrame.truncate(data, after = 100)

#获取review中字符串的最大长度
length_of_the_messages = data["review"].str.split("\\s+")
print(length_of_the_messages.str.len().max())

data_labels = pd.DataFrame(data.sentiment.values)
data_reviews = pd.DataFrame(data.review.values)
# print(type(data_labels))
# print(data_reviews)

"""
将训练数据分为两组，一组是训练集(包含80%的样本)；一组是测试集(包含20%的样本)
"""

X = data.review.values[:100]  # review
Y = data.sentiment.values[:100]  # label

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
# print(X_train)
# print(Y_train)
# print(X_test)
# print(Y_test)

"""
查看设备硬件条件，有GPU就使用GPU，否则使用CPU
"""
if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f'There are {torch.cuda.device_count()} GPU(s) available.')
    print('Device name:', torch.cuda.get_device_name(0))

else:
    print('No GPU available, using the CPU instead.')
    device = torch.device("cpu")

# 载入bert的tokenize方法
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

# 数据预处理，将数据转换为tensor
def preprocess_data(data):
    # 空列表来储存信息
    input_ids = []
    attention_masks = []

    # 每个句子循环一次
    for sent in data:
        encoded_sent = tokenizer.encode_plus(
            text=sent,
            add_special_tokens=True,  # 加入特殊字符[CLS] 和 [SEP]
            max_length=MAX_LEN,  # 截断的最大长度
            padding='max_length',  # 填充为最大长度
            return_attention_mask=True  # 返回 attention mask
        )

        # 把输出加到list里面
        input_ids.append(encoded_sent.get('input_ids'))
        attention_masks.append(encoded_sent.get('attention_mask'))

    #转换为tensor
    input_ids = torch.tensor(input_ids)
    attention_masks = torch.tensor(attention_masks)

    return input_ids, attention_masks

# Encode 准备好的数据
encoded_comment = [tokenizer.encode(sent, add_special_tokens=True) for sent in data.review.values]

#文本的最大长度
MAX_LEN = max([len(sent) for sent in encoded_comment])
# print(MAX_LEN) #MAX_len = 485

# 在训练集和测试集上运行 preprocess_data 转化为指定输入格式
train_inputs, train_masks = preprocess_data(X_train)
test_inputs, test_masks = preprocess_data(X_test)

# 转化为tensor类型

train_labels = torch.tensor(Y_train)
test_labels = torch.tensor(Y_test)


# 超参数用于微调, 个人建议batch size 16 or 32较好.
batch_size = 16

# 为训练集建立 DataLoader
train_data = TensorDataset(train_inputs, train_masks, train_labels)
train_sampler = RandomSampler(train_data)
train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)
# print(train_dataloader)

# 给测试集建立 DataLoader
test_data = TensorDataset(test_inputs, test_masks, test_labels)
test_sampler = SequentialSampler(test_data)
test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=batch_size)

print("The data is ready!")


#构建分类模型
#首先导入bert预训练模型
bert = BertModel.from_pretrained('bert-base-uncased')

#微调Bert模型，在预训练模型之后加上两个全连接层构成一个分类器，实现分类功能
class BertClassifier(nn.Module):
    def __init__(self):

        super(BertClassifier, self).__init__()
        # hidden size of Bert默认768，分类器隐藏维度，输出维度为2(因为最后的评论分为积极消极两类)
        dim_input, classify_hidden, dim_output = 768, 100, 2

        # 实体化Bert模型
        self.bert = BertModel.from_pretrained('bert-base-uncased')

        # 分类器，使用ReLU激活函数和两层全连接层
        self.classifier = nn.Sequential(
            nn.Linear(dim_input, classify_hidden),  # 全连接
            nn.ReLU(),  # 激活函数
            nn.Linear(classify_hidden, dim_output)  # 全连接
        )

    def forward(self, input_ids, attention_mask):
        # 搭建网络
        # 输入
        outputs = self.bert(input_ids=input_ids,
                            attention_mask=attention_mask)
        #提取[CLS]的最后一层的输出
        last_hidden_state_cls = outputs[0][:, 0, :]
        # 全连接，计算然后分类，最后输出label
        logits = self.classifier(last_hidden_state_cls)

        return logits

"""
初始化我们的模型，包括优化器、学习率，设置epochs
"""
def initialize_model(epochs=2):
    # 初始化分类器
    bert_classifier = BertClassifier()
    # 使用device进行运算，GPU or CPU
    bert_classifier.to(device)
    # 优化器，使用AdamW，设置学习率和eps
    optimizer = AdamW(bert_classifier.parameters(),
                      lr=5e-5,  # 默认学习率
                      eps=1e-8
                      )
    # 训练的总步数
    total_steps = len(train_dataloader) * epochs
    # 学习率预热
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=0,
                                                num_training_steps=total_steps)
    return bert_classifier, optimizer, scheduler


# 实例化损失函数，因为是分类任务，所以使用交叉熵损失函数比较好
loss_fn = nn.CrossEntropyLoss()  # 交叉熵

#训练函数
def train(model, train_dataloader, test_dataloader=None, epochs=2, evaluation=False):
    #开始训练
    for epoch_i in range(epochs):
        print(f"{'Epoch':^7} | {'每40个Batch':^9} | {'训练集 Loss':^12} | {'测试集 Loss':^10} | {'测试集准确率':^9} | {'时间':^9}")
        print("-" * 80)

        # 计算每个epoch消耗的时间
        t0_epoch, t0_batch = time.time(), time.time()

        # 每一个epoch都需要重置变量 total_loss, batch_loss, batch_counts
        total_loss, batch_loss, batch_counts = 0, 0, 0

        # 把model设置为训练模式
        model.train()

        # 分batch训练
        for step, batch in enumerate(train_dataloader):
            batch_counts += 1
            # 把batch加载到GPU
            b_input_ids, b_attn_mask, b_labels = tuple(t.to(device) for t in batch)
            # 梯度归零
            model.zero_grad()
            # 训练
            logits = model(b_input_ids, b_attn_mask)

            # 损失计算并累加
            loss = loss_fn(logits, b_labels)

            batch_loss += loss.item()
            total_loss += loss.item()
            # 反向传播
            loss.backward()
            # 归一化，防止梯度爆炸
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            # 更新参数和学习率
            optimizer.step()
            scheduler.step()

            # 输出每40个batch的消耗时间和计算损失
            if (step % 40 == 0 and step != 0) or (step == len(train_dataloader) - 1):
                # 开始计时
                time_elapsed = time.time() - t0_batch

                # 输出结果
                print(
                    f"{epoch_i + 1:^7} | {step:^10} | {batch_loss / batch_counts:^14.6f} | {'-':^12} | {'-':^13} | {time_elapsed:^9.2f}")

                # 重置batch参数
                batch_loss, batch_counts = 0, 0
                t0_batch = time.time()

        # 计算平均loss：训练集的平均loss
        avg_train_loss = total_loss / len(train_dataloader)

        print("-" * 80)

        # 可以通过控制evalution是否为TRUE，决定是否需要汇总评估
        if evaluation:
            # 每个epoch之后评估一下性能

            test_loss, test_accuracy = evaluate(model, test_dataloader)
            # 输出整个训练集上的时间消耗情况
            time_elapsed = time.time() - t0_epoch

            print(
                f"{epoch_i + 1:^7} | {'-':^10} | {avg_train_loss:^14.6f} | {test_loss:^12.6f} | {test_accuracy:^12.2f}% | {time_elapsed:^9.2f}")
            print("-" * 80)
        print("\n")

# 评价函数：在测试集上观察模型的效果
def evaluate(model, test_dataloader):

    # 把model设置为评估模式
    model.eval()

    # 测试集上准确率和误差
    test_accuracy = []
    test_loss = []

    # 测试集上的每个batch
    for batch in test_dataloader:
        # 放到GPU上
        b_input_ids, b_attn_mask, b_labels = tuple(t.to(device) for t in batch)

        # 计算结果（不更新梯度）
        with torch.no_grad():
            logits = model(b_input_ids, b_attn_mask)  # 放入模型中运行

        # 计算误差
        loss = loss_fn(logits, b_labels.long())
        test_loss.append(loss.item())

        # 得到预测结果，返回一行中最大值的序号
        preds = torch.argmax(logits, dim=1).flatten()

        # 计算准确率，预测正确的个数/测试集总的样本数
        accuracy = (preds == b_labels).cpu().numpy().mean() * 100
        test_accuracy.append(accuracy)

    # 计算整体的平均正确率和loss
    val_loss = np.mean(test_loss)
    val_accuracy = np.mean(test_accuracy)

    return val_loss, val_accuracy

print("服务器准备就绪")
bert_classifier, optimizer, scheduler = initialize_model(epochs=6)

print("开始训练:\n")
train(bert_classifier, train_dataloader, test_dataloader, epochs=6, evaluation=True)
print("训练完成！")
