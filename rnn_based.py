# RNN
# 双层
import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
from config_c import Config
import numpy as np
cfg = Config()

class CustomDataset(Dataset):
    def __init__(self, data_path, cfg):
        super().__init__()
        self.data_path = data_path
        self.word2index = cfg.word2index
        self.unk = self.word2index[cfg.unk]
        self.max_inputs_len = cfg.max_inputs_len
        with open(self.data_path, 'r', encoding='utf-8') as f:
            self.lines = f.readlines()

    def __len__(self):
        return len(self.lines)

    def __getitem__(self, item):
        line = self.lines[item]
        pairs = line.split('\t')
        label, text = pairs[0], pairs[1]
        text_index = [2] + [self.word2index.get(word, self.unk) for word in text][:self.max_inputs_len] + [3]
        return torch.tensor(text_index), torch.tensor(int(label))



class CustomAverageModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.embeddings = nn.Embedding(self.cfg.vocab_size + 2, self.cfg.embedding_dim, padding_idx=0)
        self.rnn = nn.GRU(self.cfg.embedding_dim, self.cfg.hidden_size // 2, batch_first=True, bidirectional=True, num_layers=2)
        self.ln = nn.LayerNorm(self.cfg.embedding_dim)
        self.output = nn.Linear(self.cfg.hidden_size, self.cfg.output_dim)
        self.act = nn.ReLU()
        self.dropout = nn.Dropout()
        self.init_weights()

    def init_weights(self):
        initrange = self.cfg.initrange
        self.embeddings.weight.data.uniform_(-initrange, initrange)
        self.output.weight.data.uniform_(-initrange, initrange)
        self.output.bias.data.zero_()


    def forward(self, inputs, mask):
        embedded = self.dropout(self.ln(self.embeddings(inputs)))
        rnn_outputs, rnn_hidden_state = self.rnn(embedded)
        # a = rnn_outputs * mask.unsqueeze(-1)
        average = (rnn_outputs * mask.unsqueeze(-1)).sum(1) / torch.sum(mask, dim=-1, keepdim=True)
        outputs = self.output(average)
        return torch.sigmoid(outputs.squeeze(-1))


        # inputs_length = (inputs > 0).sum(-1)
        # embedded = self.dropout(self.ln(self.embeddings(inputs)))
        # packed_embed = pack_padded_sequence(embedded, inputs_length, batch_first=True)
        # rnn_outputs, rnn_hidden_state = self.rnn(packed_embed)
        # encode_output = pad_packed_sequence(rnn_outputs, batch_first=True)[0]
        # outputs = self.output(rnn_hidden_state.squeeze(0))
        #
        # return torch.sigmoid(outputs.squeeze(-1))

def padding(batch):
    inputs, targets = zip(*batch)
    all_seq_len = [len(i) for i in inputs]
    sorted_seq = sorted(enumerate(all_seq_len), key=lambda x: x[1], reverse=True)
    sorted_inputs = tuple(inputs[i] for i, j in sorted_seq)
    sorted_targets = tuple(targets[i] for i, j in sorted_seq)
    padded_inputs = pad_sequence(sorted_inputs, batch_first=True, padding_value=0)
    inputs_mask = (padded_inputs > 0).to(torch.float32)

    return padded_inputs, inputs_mask, torch.tensor(sorted_targets, dtype=torch.float32)

def train():
    # 数据加载
    train_data = CustomDataset(cfg.train_data_path, cfg)
    train_loader = DataLoader(train_data, batch_size=cfg.batch_size, collate_fn=padding, shuffle=True)
    # 模型加载
    model = CustomAverageModel(cfg).to(cfg.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    loss_fn = torch.nn.BCELoss().to(cfg.device)
    # 迭代训练
    min_count, min_eval_loss = 0, float('inf')
    for epoch in range(cfg.max_epoch):
        total_loss, iteration = 0, 0
        model.train()
        for inputs, mask, targets in train_loader:
            inputs, mask, targets = inputs.to(cfg.device), mask.to(cfg.device), targets.to(cfg.device)
            outputs = model(inputs, mask)
            loss = loss_fn(outputs, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            iteration += 1

        eval_loss = evaluate(model, loss_fn)
        print('epoch_{}'.format(epoch))
        print('train_loss:{}'.format(total_loss/iteration))
        print('eval_loss:{}'.format(eval_loss))
        accuracy(cfg.valid_data_path, model)
        accuracy(cfg.test_data_path, model)
        print('-'*50)

        if eval_loss < min_eval_loss:
            min_eval_loss = eval_loss
            best_model = model
            min_count = 0
        else:
            min_count += 1

        if min_count == cfg.early_stop:
            break
        model_path = cfg.model_save_path + '4epoch_' + str(epoch)
        torch.save(best_model.state_dict(), model_path)

def evaluate(model, loss_fn):
    model.eval()
    valid_data = CustomDataset(cfg.valid_data_path, cfg)
    valid_loader = DataLoader(valid_data, batch_size=cfg.batch_size, collate_fn=padding)
    total_loss, iteration = 0, 0
    with torch.no_grad():
        for inputs, mask, targets in valid_loader:
            inputs, mask, targets = inputs.to(cfg.device), mask.to(cfg.device), targets.to(cfg.device)
            outputs = model(inputs, mask)
            loss = loss_fn(outputs, targets)

            total_loss += loss.item()
            iteration += 1

        return total_loss / iteration

def accuracy(data, model=None):
    # model = CustomAverageModel(cfg).to(cfg.device)
    # model.load_state_dict(torch.load('./output/4epoch_7', map_location='cpu'))
    # total_num = sum(p.numel() for p in model.parameters())
    # print('参数量:{}'.format(total_num))
    model.eval()
    test_data = CustomDataset(data, cfg)
    test_loader = DataLoader(test_data, batch_size=cfg.batch_size, collate_fn=padding)
    all_accurate, all_texts = 0, 0
    with torch.no_grad():
        for inputs, mask, targets in test_loader:
            inputs, mask, targets= inputs.to(cfg.device), mask.to(cfg.device), targets.to(cfg.device)
            outputs = model(inputs, mask)
            outputs_label = (outputs > 0.5).to(torch.float32)
            accurate = torch.sum(outputs_label == targets)
            all_accurate += accurate.item()
            all_texts += len(targets)

        print(all_accurate / all_texts)
        return all_accurate / all_texts

def text_inference():
    model = CustomAverageModel(cfg).to(cfg.device)
    model.load_state_dict(torch.load(cfg.model_load_path, map_location='cpu'))
    model.eval()

    with open(cfg.test_data_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    for line in lines:
        pairs = line.split('\t')
        label, text = pairs[0], pairs[1]
        text_index = [cfg.word2index.get(word, 1) for word in jieba.cut(text)]
        inputs = torch.tensor(text_index).unsqueeze(0)
        mask = (inputs > 0).to(torch.float32)
        outputs = model(inputs, mask)
        outputs_label = (outputs > 0.5).item()
        if outputs_label != int(label):
            print(text)
            print(label)
            print(outputs.item())
            print('-'*50)


def analysis():
    model = CustomAverageModel(cfg).to(cfg.device)
    model.load_state_dict(torch.load(cfg.model_load_path, map_location='cpu'))
    model.eval()
    embedding_weights = model.embeddings.weight.data
    U = model.attention.weight
    attention_scores = torch.cosine_similarity(embedding_weights, U, dim=-1)

    sorted_scores = sorted(enumerate(attention_scores), key=lambda x: x[1], reverse=True)
    max_scores = [cfg.index2word[str(pair[0])] for pair in sorted_scores[:15]]
    min_scores = [cfg.index2word[str(pair[0])] for pair in sorted_scores[-15:]]
    print('max：\n', max_scores)
    print('min：\n', min_scores)

if __name__ == '__main__':
    train()
    accuracy()
    # text_inference()
    # analysis()