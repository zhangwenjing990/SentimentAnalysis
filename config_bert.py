import os
import torch
from utils import load_chinese_base_vocab

class Config(object):
    def __init__(self):
        self.root_path = os.getcwd()
        # 数据存放路径
        self.train_data_path = self.root_path + '/baseline/data/train.txt'
        self.valid_data_path = self.root_path + '/baseline/data/valid.txt'
        self.test_data_path = self.root_path + '/baseline/data/test.txt'
        # 预训练模型存放路径
        self.vocab_path = self.root_path + '/bert/vocab.txt'
        self.pretrained_model_path = self.root_path + '/bert/pytorch_model.bin'
        self.save_model_path = self.root_path + '/output/classification_model.bin'

        # 一些基本配置
        # 设备
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # 序列最大长度
        self.max_seq_len = 200
        # batch大小
        self.batch_size = 8
        # 学习率
        self.learn_rate = 1e-5

        # loss打印频率
        self.print_loss_steps = 1000
        # 模型保存频率
        self.save_model_steps = 3000

        # 解码
        self.beam_size = 4
        self.max_decode_len = 20
        self.char2idx, self.keep_tokens = load_chinese_base_vocab(self.vocab_path)
        print('====')

