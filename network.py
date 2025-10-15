# -*- coding: utf-8 -*-
"""
@Time:Created on 2022/11/09 22:00
@author: Minjie Mou
@Filename: model.py
@Software: PyCharm
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import math
import numpy as np
from sklearn.metrics import roc_auc_score, precision_score, recall_score, precision_recall_curve, auc
from Radam import *
from lookahead import Lookahead
import timeit


class SelfAttention(nn.Module):
    def __init__(self, hid_dim, n_heads, dropout, device):
        super().__init__()
        self.hid_dim = hid_dim
        self.n_heads = n_heads

        assert hid_dim % n_heads == 0

        self.w_q = nn.Linear(hid_dim, hid_dim)
        self.w_k = nn.Linear(hid_dim, hid_dim)
        self.w_v = nn.Linear(hid_dim, hid_dim)

        self.fc = nn.Linear(hid_dim, hid_dim)
        self.do = nn.Dropout(dropout)
        self.scale = torch.sqrt(torch.FloatTensor([hid_dim // n_heads])).to(device)

    def forward(self, query, key, value, mask=None):
        #print('e')
        bsz = query.shape[0]
        #print('qkv',query.shape,key.shape,value.shape)
        # query = key = value [batch size, sent len, hid dim]
        Q = self.w_q(query)
        K = self.w_k(key)
        V = self.w_v(value)
        # Q, K, V = [batch size, sent len, hid dim]

        Q = Q.view(bsz, -1, self.n_heads, self.hid_dim // self.n_heads).permute(0, 2, 1, 3)
        K = K.view(bsz, -1, self.n_heads, self.hid_dim // self.n_heads).permute(0, 2, 1, 3)
        V = V.view(bsz, -1, self.n_heads, self.hid_dim // self.n_heads).permute(0, 2, 1, 3)

        # K, V = [batch size, n heads, sent len_K, hid dim // n heads]
        # Q = [batch size, n heads, sent len_q, hid dim // n heads]
        energy = torch.matmul(Q, K.permute(0, 1, 3, 2)) / self.scale
        #print('e',energy.shape)
        # energy = [batch size, n heads, sent len_Q, sent len_K]
        if mask is not None:
            energy = energy.masked_fill(mask == 0, -1e10)

        attention = self.do(F.softmax(energy, dim=-1))

        # attention = [batch size, n heads, sent len_Q, sent len_K]

        x = torch.matmul(attention, V)
        # x = [batch size, n heads, sent len_Q, hid dim // n heads]

        x = x.permute(0, 2, 1, 3).contiguous()
        # x = [batch size, sent len_Q, n heads, hid dim // n heads]

        x = x.view(bsz, -1, self.n_heads * (self.hid_dim // self.n_heads))
        # x = [batch size, src sent len_Q, hid dim]

        x = self.fc(x)
        # x = [batch size, sent len_Q, hid dim]
        #print('x',x.shape)
        return x, attention
class SelfAttention_no(nn.Module):
    def __init__(self, hid_dim, n_heads, dropout, device):
        super().__init__()
        self.hid_dim = hid_dim
        self.n_heads = n_heads

        assert hid_dim % n_heads == 0

        self.w_q = nn.Linear(hid_dim, hid_dim)
        self.w_k = nn.Linear(hid_dim, hid_dim)
        self.w_v = nn.Linear(hid_dim, hid_dim)

        self.fc = nn.Linear(hid_dim, hid_dim)
        self.do = nn.Dropout(dropout)
        self.scale = torch.sqrt(torch.FloatTensor([hid_dim // n_heads])).to(device)

    def forward(self, query, key, value, mask=None):
        bsz = query.shape[0]
        #print('qkv',query.shape,key.shape,value.shape)
        # query = key = value [batch size, sent len, hid dim]
        Q = self.w_q(query)
        K = self.w_k(key)
        V = self.w_v(value)
        # Q, K, V = [batch size, sent len, hid dim]

        Q = Q.view(bsz, -1, self.n_heads, self.hid_dim // self.n_heads).permute(0, 2, 1, 3)
        K = K.view(bsz, -1, self.n_heads, self.hid_dim // self.n_heads).permute(0, 2, 1, 3)
        V = V.view(bsz, -1, self.n_heads, self.hid_dim // self.n_heads).permute(0, 2, 1, 3)

        # K, V = [batch size, n heads, sent len_K, hid dim // n heads]
        # Q = [batch size, n heads, sent len_q, hid dim // n heads]
        energy = torch.matmul(Q, K.permute(0, 1, 3, 2)) / self.scale
        #print('e',energy.shape)
        # energy = [batch size, n heads, sent len_Q, sent len_K]
        # if mask is not None:
        #     energy = energy.masked_fill(mask == 0, -1e10)

        attention = self.do(F.softmax(energy, dim=-1))

        # attention = [batch size, n heads, sent len_Q, sent len_K]

        x = torch.matmul(attention, V)
        # x = [batch size, n heads, sent len_Q, hid dim // n heads]

        x = x.permute(0, 2, 1, 3).contiguous()
        # x = [batch size, sent len_Q, n heads, hid dim // n heads]

        x = x.view(bsz, -1, self.n_heads * (self.hid_dim // self.n_heads))
        # x = [batch size, src sent len_Q, hid dim]

        x = self.fc(x)
        # x = [batch size, sent len_Q, hid dim]
        #print('x',x.shape)
        return x, attention

class Encoder_cnn(nn.Module):
    """protein feature extraction."""

    def __init__(self, protein_dim, hid_dim, n_layers, kernel_size, dropout, device):
        super().__init__()

        assert kernel_size % 2 == 1, "Kernel size must be odd (for now)"

        self.input_dim = protein_dim
        self.hid_dim = hid_dim
        self.kernel_size = kernel_size
        self.dropout = dropout
        self.n_layers = n_layers
        self.device = device
        # self.pos_embedding = nn.Embedding(1000, hid_dim)
        self.scale = torch.sqrt(torch.FloatTensor([0.5])).to(device)
        self.convs = nn.ModuleList(
            [nn.Conv1d(hid_dim, 2 * hid_dim, kernel_size, padding=(kernel_size - 1) // 2) for _ in
             range(self.n_layers)])  # convolutional layers
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(1024, 102)
        self.fc2 = nn.Linear(1024, 102)
        self.fc3 = nn.Linear(247, 64)
        #self.fc4 = nn.Linear(64, 64)
        #self.fc3 = nn.Linear(43, self.hid_dim)
        self.gn = nn.GroupNorm(8, hid_dim * 2)
        self.ln = nn.LayerNorm(hid_dim)

    def forward(self,protein1,protein2,protein3):
        print('1233', protein1.shape, protein2.shape, protein3.shape)
        conv_input1 = self.fc1(protein1)
        conv_input2 = self.fc2(protein2)
        print('1233',conv_input1.shape,conv_input2.shape,protein3.shape)
        conv_input = torch.cat((conv_input1, conv_input2,protein3), dim=-1)
        conv_input = self.fc3(conv_input)
        print('1234',conv_input.shape)
        # conv_input=[batch size,protein len,hid dim]
        # permute for convolutional layer
        conv_input = conv_input.permute(0, 2, 1)
        # conv_input = [batch size, hid dim, protein len]
        for i, conv in enumerate(self.convs):
            print('12334')
            # pass through convolutional layer
            conved = conv(self.dropout(conv_input))
            # conved = [batch size, 2*hid dim, protein len]
            print('122234')
            # pass through GLU activation function
            conved = F.glu(conved, dim=1)
            # conved = [batch size, hid dim, protein len]
            print('122234')
            # apply residual connection / high way
            conved = (conved + conv_input) * self.scale
            # conved = [batch size, hid dim, protein len]

            # set conv_input to conved for next loop iteration
            conv_input = conved
            print('4122234')
        conved = conved.permute(0, 2, 1)
        # conved = [batch size,protein len,hid dim]
        conved = self.ln(conved)
        #conved=self.fc4(conved)
        print('123')
        return conved
class Encoder_lstm(nn.Module):
    """protein feature extraction."""

    def __init__(self, protein_dim, hid_dim, n_layers, kernel_size, dropout, device):
        super().__init__()

        assert kernel_size % 2 == 1, "Kernel size must be odd (for now)"

        self.input_dim = protein_dim
        self.hid_dim = hid_dim
        self.kernel_size = kernel_size
        self.dropout = dropout
        self.n_layers = n_layers
        self.device = device
        # self.pos_embedding = nn.Embedding(1000, hid_dim)
        self.scale = torch.sqrt(torch.FloatTensor([0.5])).to(device)
        self.convs = nn.ModuleList(
            [nn.Conv1d(hid_dim, 2 * hid_dim, kernel_size, padding=(kernel_size - 1) // 2) for _ in
             range(self.n_layers)])  # convolutional layers
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(1024, 102)
        self.fc2 = nn.Linear(1024, 102)
        self.fc3 = nn.Linear(247, 64)
        #self.fc4 = nn.Linear(64, 64)
        #self.fc3 = nn.Linear(43, self.hid_dim)
        self.gn = nn.GroupNorm(8, hid_dim * 2)
        self.ln = nn.LayerNorm(hid_dim)
        self.LSTM_cdr_encoder = nn.LSTM(
           247,
            32,
            num_layers=2,
            batch_first=True,  # 表示输入数据的第一个维度是 batch siz
            dropout=0.3,
            bidirectional=True)

    def forward(self,protein1,protein2,protein3):
        conv_input1 = self.fc1(protein1)
        conv_input2 = self.fc2(protein2)
        #print('1233',conv_input1.shape,conv_input2.shape,protein3.shape)
        conv_input = torch.cat((conv_input1, conv_input2,protein3), dim=-1)
        #conv_input = self.fc3(conv_input)
        #print('1234',conv_input.shape)
        # conv_input=[batch size,protein len,hid dim]
        # permute for convolutional layer
        print('conv1')
        x, _ = self.LSTM_cdr_encoder(conv_input)
        print('conv',x.shape)
        #x=self.fc4(x)
        return x
class Encoder_att(nn.Module):
    def __init__(self, local_dim, hid_dim, n_layers, n_heads, pf_dim, decoder_layer, self_attention,
                 positionwise_feedforward, dropout, device):
        super().__init__()
        self.ln = nn.LayerNorm(hid_dim)
        self.output_dim = local_dim
        self.hid_dim = hid_dim
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.pf_dim = pf_dim
        self.decoder_layer = decoder_layer
        self.self_attention = self_attention
        self.positionwise_feedforward = positionwise_feedforward
        self.dropout = dropout
        self.device = device
        self.sa = self_attention(hid_dim, n_heads, dropout, device)
        self.layers = nn.ModuleList(
            [decoder_layer(hid_dim, n_heads, pf_dim, self_attention, positionwise_feedforward, dropout, device)
             for _ in range(n_layers)])
        self.ft = nn.Linear(local_dim, hid_dim)
        self.do = nn.Dropout(dropout)
        self.fc1 = nn.Linear(1024, 102)
        self.fc2 = nn.Linear(1024, 102)
        self.fc3 = nn.Linear(247, 64)
        #self.fc4 = nn.Linear(64, 64)
        self.gn = nn.GroupNorm(8, 256)

    def forward(self, protein1, protein2, protein3,src_mask=None):
        conv_input1 = self.fc1(protein1)
        conv_input2 = self.fc2(protein2)
        # print('1233',conv_input1.shape,conv_input2.shape,protein3.shape)
        src = torch.cat((conv_input1, conv_input2, protein3), dim=-1)
        src = self.fc3(src)
        # trg = [batch size, local len, hid dim]

        for layer in self.layers:
            src, attention = layer(src,src_mask)
        #src=self.fc4(src)


        return src

class PositionwiseFeedforward1(nn.Module):
    def __init__(self, hid_dim, pf_dim, dropout):
        super().__init__()

        self.hid_dim = hid_dim
        self.pf_dim = pf_dim

        self.fc_1 = nn.Conv1d(hid_dim, pf_dim, 1)  # convolution neural units
        self.fc_2 = nn.Conv1d(pf_dim, hid_dim, 1)  # convolution neural units

        self.do = nn.Dropout(dropout)

    def forward(self, x):
        # x = [batch size, sent len, hid dim]

        x = x.permute(0, 2, 1)
        # x = [batch size, hid dim, sent len]

        x = self.do(F.relu(self.fc_1(x)))
        # x = [batch size, pf dim, sent len]

        x = self.fc_2(x)
        # x = [batch size, hid dim, sent len]

        x = x.permute(0, 2, 1)
        # x = [batch size, sent len, hid dim]

        return x

class PositionwiseFeedforward2(nn.Module):
    def __init__(self, hid_dim, pf_dim, dropout):
        super().__init__()

        self.hid_dim = hid_dim
        self.pf_dim = pf_dim

        self.fc_1 = nn.Conv1d(hid_dim, pf_dim, 1)  # convolution neural units
        self.fc_2 = nn.Conv1d(pf_dim, hid_dim, 1)  # convolution neural units

        self.do = nn.Dropout(dropout)

    def forward(self, x):
        # x = [batch size, sent len, hid dim]

        x = x.permute(0, 2, 1)
        # x = [batch size, hid dim, sent len]

        x = self.do(F.relu(self.fc_1(x)))
        # x = [batch size, pf dim, sent len]

        x = self.fc_2(x)
        # x = [batch size, hid dim, sent len]

        x = x.permute(0, 2, 1)
        # x = [batch size, sent len, hid dim]

        return x
class DecoderLayer1(nn.Module):
    def __init__(self, hid_dim, n_heads, pf_dim, self_attention, positionwise_feedforward, dropout, device):
        super().__init__()

        self.ln = nn.LayerNorm(hid_dim)
        self.sa = self_attention(hid_dim, n_heads, dropout, device)
        self.ea = self_attention(hid_dim, n_heads, dropout, device)
        self.pf = positionwise_feedforward(hid_dim, pf_dim, dropout)
        self.do = nn.Dropout(dropout)
        self.proj = nn.Sequential(
            nn.Linear(hid_dim * 3, 1, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, src, src_mask=None):
        src_2 = src
        #print('scr', src.shape)
        src, attention = self.sa(src, src, src, src_mask)
        #print('scr',src.shape)
        gate_input = torch.cat((src, src_2, src - src_2), dim=-1)
        gate = self.proj(gate_input)
        gate_out = src * gate + src_2 * (1 - gate)
        src = self.ln(gate_out)
        # src = self.ln(trg_2 + self.do(src))

        src_1 = src
        src, attention = self.ea(src, src, src, src_mask)
        gate_input = torch.cat((src, src_1, src - src_1), dim=-1)
        gate = self.proj(gate_input)
        gate_out = src * gate + src_1 * (1 - gate)
        src11 = self.ln(gate_out)

        trg_3 = src11
        trg = self.ln(trg_3 + self.do(self.pf(src11)))

        return trg, attention

class DecoderLayer2(nn.Module):
    def __init__(self, hid_dim, n_heads, pf_dim, self_attention, positionwise_feedforward, dropout, device):
        super().__init__()

        self.ln = nn.LayerNorm(hid_dim)
        self.sa = self_attention(hid_dim, n_heads, dropout, device)
        self.ea = self_attention(hid_dim, n_heads, dropout, device)
        self.pf = positionwise_feedforward(hid_dim, pf_dim, dropout)
        self.do = nn.Dropout(dropout)
        self.proj = nn.Sequential(
            nn.Linear(hid_dim * 3, 1, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, src1,src2,src3, src_mask=None):
        src_222 = src1
        #print('scr', src.shape)
        src, attention = self.sa(src1,src2,src3, src_mask)
        #print('scr',src.shape)
        gate_input = torch.cat((src, src_222, src - src_222), dim=-1)
        gate = self.proj(gate_input)
        gate_out = src * gate + src_222 * (1 - gate)
        src = self.ln(gate_out)
        # src = self.ln(trg_2 + self.do(src))

        src_1 = src
        src, attention = self.ea(src, src, src, src_mask)
        gate_input = torch.cat((src, src_1, src - src_1), dim=-1)
        gate = self.proj(gate_input)
        gate_out = src * gate + src_1 * (1 - gate)
        src11 = self.ln(gate_out)

        trg_3 = src11
        trg = self.ln(trg_3 + self.do(self.pf(src11)))

        return trg,trg,trg, attention


class Decoder(nn.Module):
    def __init__(self, local_dim, hid_dim, n_layers, n_heads, pf_dim, decoder_layer, self_attention,
                 positionwise_feedforward, dropout, device):
        super().__init__()
        self.ln = nn.LayerNorm(hid_dim)
        self.output_dim = local_dim
        self.hid_dim = hid_dim
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.pf_dim = pf_dim
        self.decoder_layer = decoder_layer
        self.self_attention = self_attention
        self.positionwise_feedforward = positionwise_feedforward
        self.dropout = dropout
        self.device = device
        self.sa = self_attention(hid_dim, n_heads, dropout, device)
        self.layers = nn.ModuleList(
            [decoder_layer(hid_dim, n_heads, pf_dim, self_attention, positionwise_feedforward, dropout, device)
             for _ in range(n_layers)])
        self.ft = nn.Linear(hid_dim, hid_dim)
        self.do = nn.Dropout(dropout)
        self.fc_1 = nn.Linear(64, 256)
        self.fc_2 = nn.Linear(256, 64)
        self.fc_3 = nn.Linear(64, 2)
        self.fc_4 = nn.Linear(192, 64)
        self.gn = nn.GroupNorm(8, 256)
        self.ln = nn.LayerNorm(hid_dim)

    def forward(self,enc_src1,enc_src2,enc_src3, src_mask=None):
        # trg = [batch_size, local len, local_dim]
        # src = [batch_size, protein len, hid_dim] # encoder output
        #src = self.ft(src)
        # trg = [batch size, local len, hid dim]
        #src111=src1
        # src = torch.cat((enc_src1,enc_src2,enc_src3), dim=-1)
        # enc_src = self.fc_4(src)
        for layer in self.layers:
            enc_src1,enc_src2,enc_src3, attention = layer(enc_src1,enc_src2,enc_src3,  src_mask)
        #src=src1+src111
        # trg = [batch size, local len, hid dim]
        # attention=1
        """Use norm to determine which atom is significant. """
        norm = torch.norm(enc_src1, dim=2)
        #print('norm', norm.shape)
        # norm = [batch size,local len]
        norm = F.softmax(norm, dim=1)
        # print('norm', norm.shape, norm)
        # norm = [batch size,local len]
        # trg = torch.squeeze(trg,dim=0)
        # norm = torch.squeeze(norm,dim=0)
        sum = torch.zeros((enc_src1.shape[1], 64)).to(self.device)
        for i in range(norm.shape[1]):
            v = enc_src1[0, i, :]
            # print('v1',v.shape)
            v = v * norm[:, i]
            # print('v2',v.shape)
            sum[i,] += v

        label = F.relu(self.fc_1(sum))
        label = self.do(label)
        label = F.relu(self.fc_2(label))
        label = self.fc_3(label)
        # print('lab',label)
        #return sum, label
        return sum, attention, label


class Predictor(nn.Module):
    def __init__(self, encoder1, encoder2, encoder3, decoder, device):
        super().__init__()
        self.encoder1 = encoder1
        self.encoder2 = encoder2
        self.encoder3 = encoder3
        self.decoder = decoder
        self.device = device

    def make_masks(self,  protein_max_len):
        #N = len(local_num)  # batch size
        #local_mask = torch.zeros((N, local_max_len))
        protein_mask = torch.zeros((protein_max_len, protein_max_len))
        for i in range(protein_max_len):
            #local_mask[i, :local_num[i]] = 1
            protein_mask[i, :i] = 1
        #local_mask = local_mask.unsqueeze(1).unsqueeze(3).to(self.device)
        protein_mask = protein_mask.unsqueeze(0).unsqueeze(1).to(self.device)
        #print('mask',protein_mask.shape)
        return  protein_mask

    def forward(self, protein1,protein2,protein3):
        # local = [batch,local_num, local_dim]
        # protein = [batch,protein len, 100]

        protein_max_len = protein1.shape[1]
        protein_mask = self.make_masks( protein_max_len)

        enc_src1 = self.encoder1(protein1,protein2,protein3)
        enc_src2 = self.encoder2(protein1, protein2, protein3)
        enc_src3 = self.encoder3(protein1, protein2, protein3)
        # enc_src = [batch size, protein len, hid dim]

        sum, attention, out = self.decoder.forward(enc_src1,enc_src2,enc_src3,  protein_mask)

        return sum, attention, out

    def __call__(self, data, train=None, valid=None):

        Loss = nn.CrossEntropyLoss(weight=torch.from_numpy(np.array([1, 5])).float().to(self.device))

        if train:
            protein1,protein2,protein3,index,correct_interaction= data
            sum, attention, predicted_interaction = self.forward(protein1,protein2,protein3)
            #print('1qw',correct_interaction)
            #print('2qw',predicted_interaction)
            rounded_target = torch.round(correct_interaction)
            # 将浮点数转换为 Long 类型张量
            correct_interaction = rounded_target.type(torch.long)
            #print('2qw', predicted_interaction)
            loss2 = Loss(predicted_interaction, correct_interaction)
            return loss2

        elif valid:
            protein1, protein2, protein3, index, labels = data
            sum, attention, predicted_interaction = self.forward(protein1, protein2, protein3
                                                                 )
            labels = torch.round(labels)
            labels = labels.type(torch.long)
            loss2 = Loss(predicted_interaction, labels)
            ys = F.softmax(predicted_interaction, 1).to('cpu').data.numpy()
            predicted_labels = np.argmax(ys, axis=1)
            predicted_scores = ys[:, 1]

            return labels, predicted_labels, predicted_scores, loss2
        else:
            protein1, protein2, protein3, index = data
            sum, attention, predicted_interaction = self.forward(protein1, protein2, protein3
                                                                 )
            ys = F.softmax(predicted_interaction, 1).to('cpu').data.numpy()
            predicted_labels = np.argmax(ys, axis=1)
            predicted_scores = ys[:, 1]
            return predicted_labels, predicted_scores


def todevice(protein1,protein2,protein3,index,label, device):
    # locals_new = torch.Tensor(locals).to(device)
    print(protein1)
    proteins1 = torch.Tensor(protein1).to(device)
    proteins2 = torch.Tensor(protein2).to(device)
    proteins3 = torch.Tensor(protein3).to(device)
    index = index
    labels = np.array(label)
    # 将 NumPy 数组转换为 PyTorch tensor
    labels_new = torch.from_numpy(labels).to(device)


    return ( proteins1, proteins2,proteins3,index,labels_new)


class Trainer(object):
    def __init__(self, model, lr, weight_decay):
        self.model = model
        # w - L2 regularization ; b - not L2 regularization
        weight_p, bias_p = [], []

        # for p in self.model.parameters():
        #     if p.dim() > 1:
        #         nn.init.xavier_uniform_(p)
        #     print('p', p)
        for name, p in self.model.named_parameters():
            if 'bias' in name:
                bias_p += [p]
            else:
                weight_p += [p]
            #print('pp', p)
        # self.optimizer = optim.Adam([{'params': weight_p, 'weight_decay': weight_decay}, {'params': bias_p, 'weight_decay': 0}], lr=lr)
        self.optimizer_inner = RAdam(
            [{'params': weight_p, 'weight_decay': weight_decay}, {'params': bias_p, 'weight_decay': 0}], lr=lr)
        self.optimizer = Lookahead(self.optimizer_inner, k=5, alpha=0.5)

    def train(self, dataloader, device):
        self.model.train()
        # np.random.shuffle(dataset)
        loss_total = 0
        i = 0
        self.optimizer.zero_grad()
        # tt=0
        protein1 = dataloader[0]
        protein2= dataloader[1]
        protein3= dataloader[2]
        index=dataloader[3]
        label = dataloader[4]
        # print('1',local_num)
        # print('2', protein_num)
        data_pack = todevice(protein1,protein2,protein3,index,label, device)
        loss = self.model(data_pack, train=True)
        # print(loss)
        # loss = loss1 + loss2
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        # tt+=1
        loss_total += loss.item()

        return loss_total


class Valid1(object):
    def __init__(self, model):
        self.model = model

    def valid(self, dataloader, device):
        self.model.eval()
        T, Y, S = [], [], []
        # np.random.shuffle(dataset)
        loss_total = 0
        i = 0
        # tt=0
        with torch.no_grad():
            protein1 = dataloader[0]
            protein2 = dataloader[1]
            protein3 = dataloader[2]
            index = dataloader[3]
            label = dataloader[4]
            # print('1',local_num)
            # print('2', protein_num)
            data_pack = todevice(protein1, protein2, protein3, index, label, device)
            correct_labels, predicted_labels, predicted_scores, loss = self.model(data_pack, valid=True)
            loss_total += loss.item()
            T.extend(correct_labels)
            Y.extend(predicted_labels)
            S.extend(predicted_scores)

        return loss_total, T, Y, S


class Tester(object):
    def __init__(self, model):
        self.model = model

    def test(self, dataloader, device):
        self.model.eval()
        T, Y, S = [], [], []
        with torch.no_grad():
            protein1 = dataloader[0]
            protein2 = dataloader[1]
            protein3 = dataloader[2]
            index = dataloader[3]
            label = dataloader[4]
            # print('1',local_num)
            # print('2', protein_num)
            data_pack = todevice(protein1, protein2, protein3, index, label, device)
            correct_labels, predicted_labels, predicted_scores = self.model(data_pack, train=False, valid=False)
            T.extend(correct_labels)
            Y.extend(predicted_labels)
            S.extend(predicted_scores)
        return T, Y, S

    def save_AUCs(self, AUCs, filename):
        with open(filename, 'a') as f:
            f.write('\t'.join(map(str, AUCs)) + '\n')

    def save_model(self, model, filename):
        torch.save(model.module.state_dict(), filename)


class Predictor_test(object):
    def __init__(self, model):
        self.model = model

    def test(self, dataloader, device):
        self.model.eval()
        Y, S = [], []
        with torch.no_grad():
            protein1 = dataloader[0]
            protein2 = dataloader[1]
            protein3 = dataloader[2]
            protein1 = torch.Tensor(protein1).to(device)
            print('1', type(protein2))
            protein2 = torch.Tensor(protein2).to(device)
            protein3 = torch.Tensor(protein3).to(device)
            index = dataloader[3]
            data_pack = (protein1,protein2,protein3, index)
            # print('1',local_num)
            # print('2', protein_num)
            #data_pack = todevice(protein1, protein2, protein3, index, device)
            #proteins_new = torch.Tensor(protein).to(device)
            predicted_labels, predicted_scores = self.model(data_pack, train=False,valid=False)
            #print('3',precision_score,predicted_labels)
            Y.extend(predicted_labels)
            S.extend(predicted_scores)
            #print('11YS',Y,S)
        return Y, S
