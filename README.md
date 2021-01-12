# summarization

本项目详细的文档总结地址：https://shimo.im/docs/cqdVGQQhdCqgtG9J/ 《情感分析项目实战》

## 任务描述

​	文本情感分析(Sentiment Analysis)是指利用自然语言处理技术，对带有情感色彩的主观性文本进行分析。

​	情感分析是文本分类的一种，主要是判断一段文本的情感倾向属于正面，还是负面。现如今，随着电商平台的崛起，人们习惯于对所购买物品进行评价。通过分析用户评论，了解用户喜好，有助于商家对商品进行优化提升，改善营销和服务策略。

​	本项目针对电商平台的评论文本，分析文本的情感倾向。

## 数据描述

本项目采用online_shopping_10_cats数据集，地址：<https://github.com/SophonPlus/ChineseNlpCorpus/blob/master/datasets/online_shopping_10_cats/intro.ipynb>

该数据集包含6万多条电商平台评论文本，其中正向评论31728条；负向评论31046条。本文按8：1：1分为训练集，验证集，测试集。

## 评价指标

本项目以分类准确率为评估指标。



## 模型结果

baseline模型：0.9018

attention加权模型：0.9049

rnn模型：0.92752

bert模型：0.94233