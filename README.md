# nlp_work
sentiment_analysis
自然语言处理大作业
| 硬件环境（CPU/GPU）：  GPU                                   |                                                              |
| :----------------------------------------------------------- | ------------------------------------------------------------ |
| 操作系统：  Windows10                                        |                                                              |
| 采用的深度学习框架、工具、语言：  深度学习框架：PyTorch  工具：Jupyter Notebook、PyCharm  语言：python |                                                              |
| 任务描述/问题定义：（限200字）  任务描述：对电影评论进行情感分析，判断该评论是积极的还是消极的。  问题定义：随着最近十几年的电影行业的发展，各种类型的良莠不一的电影越来越多，但如今社会生活节奏快，导致人们并没有足够的时间去鉴赏全部的电影，通过部分观众的影评，分析其中的情感倾向，来判断电影的质量优劣，进而帮助其他观众进行观影选择。因此，从大量的电影评论中分析出观众的情感倾向具有很好的现实意义。 |                                                              |
| 数据集及来源（相关链接）：  IMDB数据集：http://ai.stanford.edu/~amaas/data/sentiment/ |                                                              |
| 采用的深度学习模型：  BERT 模型  (BidirectionalEncoder Representations from  Transformer) | 模型提出的年份及发表的会议/期刊：  提出年份：2018年10月  发表的会议：NAACL 2019 |
| 最终设置的超参数（如Learning rate, Batch size等）：  Batch  size = 16  Learning  rate = 5e-5  eps=1e-8  epoch  = 6 |                                                              |
| 模型的效果（如准确率等与任务相关的评价指标）：  训练6个epoch以后：训练集上的Loss为：0.009002；测试集上的Loss为：0.502870；测试集准确率为：91.95%。 |                                                              |

**1.**   **深度学习框架、工具等的安装及配置过程**

------

**1.1 深度学习框架的安装与配置**

Anaconda的安装与配置：进入Anaconda官网进行下载，按照步骤进行安装，然后配置环境。(可以参照网上教程进行安装)。

在Anaconda中创建环境：`conda create -n nlpwork python==3.8.13`，成功创建python环境如图所示。

![](https://github.com/light6them6up/nlp_work/blob/main/images/image-20221214132724147.png)

输入指令激活已经创建的python环境：`activate nlpwork`，然后输入：`conda list`查看已经安装的python库。python库主要包括pandas、transformers、torch、torchtext、scikit-learn等，激活环境并查看安装的python库如图所示。相关python库的安装使用：`conda install package_name`,如果网速不行，可以使用清华源等国内的镜像源。

![](https://github.com/light6them6up/nlp_work/blob/main/images/image-20221214132818395.png)

**1.2 安装torchtext库**

在安装torchtext库时，因为torchtext库因为版本不同的问题导致某些函数的所在位置可能会发生变化。所以尽可能安装torchtext库的版本为0.9.0。如图所示。

![](https://github.com/light6them6up/nlp_work/blob/main/images/image-20221214132949367.png)
**1.3 安装 spaCy**

spaCy库是一个NLP领域的文本预处理Python库，包括分词(Tokenization)、词性标注(Part-of-speech Tagging, POS Tagging)、依存分析(Dependency Parsing)、词形还原(Lemmatization)、句子边界检测(Sentence Boundary Detection，SBD)、命名实体识别(Named Entity Recognition, NER)等功能。spaCy具有简单易用，支持多种语言、使用深度学习模型进行分词等任务、提供不同尺寸的模型可供选择。

输入：`pip install spacy -i https://pypi.tuna.tsinghua.edu.cn/simple`进行安装。

**2.**  **采用的模型的详细描述**

------

**2.1 BERT模型简介**

BERT是2018年10月由Google AI研究院提出的一种预训练模型，全称是Bidirectional Encoder Representation from Transformers。BERT在机器阅读理解顶级水平测试SQuAD1.1中表现出惊人的成绩: 全部两个衡量指标上全面超越人类，并且在11种不同NLP测试中创出SOTA表现，包括将GLUE基准推高至80.4% (绝对改进7.6%)，MultiNLI准确度达到86.7% (绝对改进5.6%)，成为NLP发展史上的里程碑式的模型成就。首次提出BERT模型的论文BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding发表在自然语言处理顶会NAACL上，并获得了该届会议的最佳长论文。BERT的基础建立在transformer之上，拥有强大的语言表征能力和特征提取能力。BERT模型在GLUE基准测试中的效果见下表。

| System           | MNLI-(m/mm)  392k | QQP  363k | QNLI  108k | SST-2  67k | CoLA  8.5k | STS-B  5.7k | MRPC  3.5k | RTE  2.5k | Average |
| ---------------- | ----------------- | --------- | ---------- | ---------- | ---------- | ----------- | ---------- | --------- | :------ |
| Pre-OpenAl SOTA  | 80.6/80.1         | 66.1      | 82.3       | 93.2       | 35.0       | 81.0        | 86.0       | 61.7      | 74.0    |
| BiLSTM+ELMo+Attn | 76.4/76.1         | 64.8      | 79.8       | 90.4       | 36.0       | 73.3        | 84.9       | 56.8      | 71.0    |
| OpenAI GPT       | 82.1/81.4         | 70.3      | 87.4       | 91.3       | 45.4       | 80.0        | 82.3       | 56.0      | 75.1    |
| BERTBASE         | 84.6/83.4         | 71.2      | 90.5       | 93.5       | 52.1       | 85.8        | 88.9       | 66.4      | 79.6    |
| BERTLARGE        | 86.7/85.9         | 72.1      | 92.7       | 94.9       | 60.5       | 86.5        | 89.3       | 70.1      | 82.1    |

**2.2 模型结构**

BERT模型是一个双向编码模型，采用了Transformer Encoder block进行连接，最后在BERT后面连接上特定任务的分类器。BERT模型结构可以说就是Transformer的encoder部分，BERT模型有两种，一种叫BERT-base对应的是12层encoder，另一种叫做BERT-large对应的是24层encoder，两者主要是在模型复杂度上不同，BERT-base总的参数量为110M，BERT-large总的参数量为340M。

BERT是基于Transformer的双向预训练语言模型。模型的使用分为两个部分：pre-training(预训练)和fine-tuning(微调)。在pre-training过程中，模型将使用无标签的数据(unlabled data)来对BERT的两个预训练任务(Masked LM and NSP)进行训练。在fine-tuning过程中，BERT模型将用预训练模型初始化所有参数，这些参数将针对于下游任务，如分类任务等，使用labeled data进行训练。针对不同的下游任务可以训练出不同的模型，但是它们都是由同一个预训练模型进行初始化而来的。BERT模型除了输出层外，在预训练和微调中使用了相同的架构，相同的预训练模型参数用于初始化不同的下游任务的模型，在微调中，对所有参数进行微调。如图所示为BERT模型的预训练和微调的架构。

![](https://github.com/light6them6up/nlp_work/blob/main/images/image-20221214133321820.png)

BERT模型使用示例如下图所示。

![](https://github.com/light6them6up/nlp_work/blob/main/images/image-20221214133338229.png)

**2.3 模型的输入和输出**

BERT模型的输入的特殊之处在于，在一句话最开始拼接了一个[CLS] token，如图所示。这个特殊的[CLS] token经过BERT得到的向量表示通常被用作当前的句子表示。除此之外，其余输入的特殊单词类似Transformer。BERT模型将一串单词作为输入，这些单词在多层encoder中不断向上流动，每一层都会经过自注意力机制和前馈神经网络。BERT模型输入的文本是在进行token时是有自己的方式的，使用的是WordPieces作为最小的处理单元，语句处理成tokens后就需要考虑位置编码以及tokens, CLS，SEP进行Embedding编码。与Transformer不同的地方在于位置编码的使用，Transformer中的位置编码使用三角函数进行表示，而BERT模型的位置编码将在预训练过程中得到。

![](https://github.com/light6them6up/nlp_work/blob/main/images/image-20221214133443553.png)

BERT的输入表示如下图所示。

![](https://github.com/light6them6up/nlp_work/blob/main/images/image-20221214133458632.png)

BERT模型输入的所有token经过编码以后，会在每个位置输出一个大小为hidden_size的向量。原论文中在文本分类任务中，在获取[CLS]对应的向量以后，连接上一个分类器就可以进行分类。如图所示为单个句子文本分类任务的实现。BERT模型不仅可以使用最后一层的BERT的输出，还可以使用每一个encoder layer的每一个token的向量作为特征。比如，直接提取每一个encoder的token表示当作特征，输入现有的特定任务神经网络中进行训练。如下图所示。

![](https://github.com/light6them6up/nlp_work/blob/main/images/image-20221214133514920.png)

![](https://github.com/light6them6up/nlp_work/blob/main/images/image-20221214133538238.png)

**2.4 模型的预训练过程**

BERT模型预训练的过程使用的是Masked language model(掩码语言模型)。通过将输入文本序列的部分单词随机遮掩掉(即Mask掉)，然后让模型去预测这些被Mask的单词。如图2.7所示为BERT的掩码语言模型架构。这样做的好处在于可以加入噪声，然后再加入噪声以后进行训练可以让模型更加的稳定。但是这也带来了模型收敛较慢的后果。

![](https://github.com/light6them6up/nlp_work/blob/main/images/image-20221214133611547.png)

**2.5 损失函数**

考虑到所要完成的任务为二分类或者多分类任务，所以损失函数选择使用交叉熵损失函数。交叉熵是信息论中的一个重要的概念，主要应用于度量两个概率分布之间的差异性。交叉熵能够衡量同一个随机变量中的两个不同概率分布的差异程度，在机器学习中就表示为真实概率分布与预测概率分布之间的差异。交叉熵的值越小，模型预测效果就越好。计算公式见下式。

![](https://github.com/light6them6up/nlp_work/blob/main/images/GONGSHI.png)

式中$p(xi)$ 表示样本的真实分布，$q(xi)$表示模型所预测的分布。

 ## 运行结果：


![](https://github.com/light6them6up/nlp_work/blob/main/images/%E8%BF%90%E8%A1%8C%E7%BB%93%E6%9E%9C.png)

