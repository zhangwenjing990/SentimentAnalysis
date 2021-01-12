# !/usr/bin/env Python
# coding=utf-8
# 重写bert模型
# 重新使用源代码的tokenize方式
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from config_bert import Config

from glob import glob
from tqdm import tqdm

from utils import load_chinese_base_vocab, load_pretrained_bert, load_custom_model, Tokenizer
from model import ClassificationModel

cfg = Config()
tokenizer = Tokenizer(cfg.char2idx)


class CustomDataset(Dataset):
    # 自定义数据集加载方式
    def __init__(self, data_path, tokenizer, cfg):
        super().__init__()
        self.data_path = data_path
        self.tokenizer = tokenizer
        self.cfg = cfg
        with open(self.data_path, 'r', encoding='utf-8') as f:
            self.lines = f.readlines()


    def __len__(self):
        return len(self.lines)

    def __getitem__(self, item):
        line = self.lines[item]
        pairs = line.split('\t')
        label, text = pairs[0], pairs[1]

        input_index, _ = self.tokenizer.encode(text, max_length=self.cfg.max_seq_len)

        return torch.tensor(input_index), torch.tensor(int(label))

def padding(batch):
    inputs, targets = zip(*batch)
    padded_inputs = pad_sequence(inputs, batch_first=True, padding_value=0)
    inputs_mask = (padded_inputs > 0).to(torch.float32)

    return padded_inputs, inputs_mask, torch.tensor(targets)

def train():
    # 加载数据
    tokenizer = Tokenizer(cfg.char2idx)
    train_dataset = CustomDataset(cfg.train_data_path, tokenizer, cfg)
    train_dataloader = DataLoader(train_dataset, batch_size=cfg.batch_size, collate_fn=padding,
                                  shuffle=True, num_workers=4, pin_memory=True)
    model = ClassificationModel(len(cfg.char2idx))
    # model = load_pretrained_bert(model, cfg.pretrained_model_path, keep_tokens=cfg.keep_tokens).to(cfg.device)
    model = load_custom_model(model, cfg.save_model_path).to(cfg.device)

    loss_function = nn.CrossEntropyLoss().to(cfg.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learn_rate)
    # 迭代训练
    iteration, train_loss = 0, 0
    model.train()
    for inputs, mask, targets in tqdm(train_dataloader, position=0, leave=True):
        inputs, mask, targets = inputs.to(cfg.device), mask.to(cfg.device), targets.to(cfg.device)
        prediction = model(inputs, mask)
        loss = loss_function(prediction, targets.reshape(-1))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        iteration += 1

        if iteration % cfg.print_loss_steps == 0:
            eval_loss = evaluate(model, tokenizer, loss_function)
            print('')
            print('train_loss:{}'.format(train_loss/cfg.print_loss_steps))
            print('evalu_loss:{}'.format(eval_loss))
            accuracy(model, tokenizer, cfg.valid_data_path)
            accuracy(model, tokenizer, cfg.test_data_path)
            model.train()
            train_loss = 0

        if iteration % cfg.save_model_steps == 0:
            torch.save(model.state_dict(), cfg.save_model_path)


def evaluate(model, tokenizer, loss_function):
    # 加载验证集验证
    eval_dataset = CustomDataset(cfg.valid_data_path, tokenizer, cfg)
    eval_dataloader = DataLoader(eval_dataset, batch_size=cfg.batch_size, collate_fn=padding,
                                  shuffle=True, num_workers=4, pin_memory=True)
    model.eval()
    iteration, eval_loss = 0, 0
    with torch.no_grad():
        for inputs, mask, targets in eval_dataloader:
            inputs, mask, targets = inputs.to(cfg.device), mask.to(cfg.device), targets.to(cfg.device)
            prediction = model(inputs, mask)
            loss = loss_function(prediction, targets.reshape(-1))
            eval_loss += loss.item()
            iteration += 1
        return eval_loss / iteration



def accuracy(model, tokenizer, data):
    # 加载验证集验证
    test_dataset = CustomDataset(data, tokenizer, cfg)
    eval_dataloader = DataLoader(test_dataset, batch_size=cfg.batch_size, collate_fn=padding,
                                  shuffle=True, num_workers=4, pin_memory=True)
    model.eval()
    accurate, total_txts = 0, 0
    with torch.no_grad():
        for inputs, mask, targets in eval_dataloader:
            inputs, mask, targets = inputs.to(cfg.device), mask.to(cfg.device), targets.to(cfg.device)
            prediction = model(inputs, mask)
            total_txts += len(targets)
            accurate += torch.sum(prediction.argmax(-1) == targets).item()
        print(accurate / total_txts)

def inference_all():
    # 加载验证集验证
    model = ClassificationModel(len(cfg.char2idx))
    model = load_custom_model(model, cfg.save_model_path).to(cfg.device)

    tokenizer = Tokenizer(cfg.char2idx)
    error = 0
    with open(cfg.test_data_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    for line in lines:
        pairs = line.split('\t')
        label, text = pairs[0], pairs[1]
        input_index, _ = tokenizer.encode(text, max_length=cfg.max_seq_len)
        inputs = torch.tensor(input_index).unsqueeze(0)
        inputs_mask = (inputs > 0).to(torch.float32)
        with torch.no_grad():
            scores = model(inputs, inputs_mask)
            prediction = scores.argmax(-1).item()
        if prediction != int(label):
            print(scores[:,int(label)].item())
            print(label)
            print(text)
            print('-'*50)
            error += 1
    print(error)

def inference_random():
    # 加载验证集验证
    model = ClassificationModel(len(cfg.char2idx))
    model = load_custom_model(model, cfg.save_model_path).to(cfg.device)

    tokenizer = Tokenizer(cfg.char2idx)
    error = 0
    with open(cfg.test_data_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    for line in lines:
        pairs = line.split('\t')
        label, text = pairs[0], pairs[1]
        input_index, _ = tokenizer.encode(text, max_length=cfg.max_seq_len)
        inputs = torch.tensor(input_index).unsqueeze(0)
        inputs_mask = (inputs > 0).to(torch.float32)
        with torch.no_grad():
            scores = model(inputs, inputs_mask)
            prediction = scores.argmax(-1).item()
        if prediction != int(label):
            print(scores[:,int(label)].item())
            print(label)
            print(text)
            print('-'*50)
            error += 1
    print(error)

if __name__ == '__main__':
    # train()
    inference()
