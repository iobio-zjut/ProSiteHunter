# -*- coding: utf-8 -*-
"""
@Time:Created on 2022/11/12 10:00
@author: Minjie Mou
@Filename: main.py
@Software: PyCharm
"""
import torch
import numpy as np
import sys
import random
import os
import torch.utils.data.sampler as sampler
from network import *
import pickle
import timeit
from sklearn.utils import shuffle
from sklearn.metrics import confusion_matrix, roc_auc_score, precision_score, recall_score, roc_curve, \
    precision_recall_curve, auc, accuracy_score, matthews_corrcoef
import pandas as pd
from dataProcessingUtils_new import *
import argparse
from sklearn.decomposition import PCA

map_res = {
    'A': 'ALA',
    'R': 'ARG',
    'N': 'ASN',
    'D': 'ASP',
    'C': 'CYS',
    'Q': 'GLN',
    'E': 'GLU',
    'G': 'GLY',
    'H': 'HIS',
    'I': 'ILE',
    'L': 'LEU',
    'K': 'LYS',
    'M': 'MET',
    'F': 'PHE',
    'P': 'PRO',
    'S': 'SER',
    'T': 'THR',
    'W': 'TRP',
    'Y': 'TYR',
    'V': 'VAL',
    'Z': 'GLU',
    'B': 'ASP',
    'X': 'UNK'
}
def init_seeds(seed=0, cuda_deterministic=True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if cuda_deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True


def metrics(correct_labels, predicted_labels, predicted_scores):
    ACC = accuracy_score(correct_labels, predicted_labels)
    AUC = roc_auc_score(correct_labels, predicted_scores)
    CM = confusion_matrix(correct_labels, predicted_labels)
    TN = CM[0][0]
    FP = CM[0][1]
    FN = CM[1][0]
    TP = CM[1][1]
    Rec = TP / (TP + FN)
    Pre = TP / (TP + FP)
    F1 = 2 * Pre * Rec / (Pre + Rec)
    MCC = matthews_corrcoef(correct_labels, predicted_labels)
    precision, recall, _ = precision_recall_curve(correct_labels, predicted_scores)
    PRC = auc(recall, precision)

    # 计算特定性（Specificity）
    Specificity = TN / (TN + FP)

    return ACC, AUC, Rec, Pre, F1, MCC, PRC, Specificity


def read_fasta(file_path):
    """
    读取FASTA文件并返回序列名和序列的列表
    """
    sequences = []
    headers = []
    with open(file_path, 'r') as f:
        current_header = ""
        current_seq = []
        for line in f:
            line = line.strip()
            if line.startswith('>'):
                if current_header:  # 保存前一条序列
                    headers.append(current_header)
                    sequences.append(''.join(current_seq))
                    current_seq = []
                current_header = line[1:]  # 去掉'>'符号
            else:
                current_seq.append(line)
        # 添加最后一条序列
        if current_header:
            headers.append(current_header)
            sequences.append(''.join(current_seq))
    return headers, sequences

def feature2(file):
    csv_reader = csv.reader(file)
    # 跳过第一行
    next(csv_reader)
    dict1={}
    # 逐行读取数据并提取第1、4、6列数据
    for row in csv_reader:
        if len(row) >= 6:  # 检查行长度是否足够
            # print(row)
            data1 = row[0][1:]  # 第1列数据
            data2 = row[3]  # 第4列数据
            data3 = row[10]  # 第4列数据
            data4 = row[11]  # 第4列数据
            data5 = row[12]  # 第4列数据data4 = row[3]  # 第4列数据
            data8 = row[13]  # 第4列数据
            data9 = row[14]  # 第4列数据
            data10 = row[15]  # 第4列数据data4 = row[3]  # 第4列数据
            data11 = row[16]  # 第4列数据
            data12 = row[17]  # 第4列数据
            data6 = row[18]  # 第4列数据
            data7 = row[19]  # 第4列数据
            if data1 in dict1:
                dict1[data1].append((data2, data3, data4, data5, data8, data9, data10, data11, data12, data6, data7))
            else:
                dict1[data1] = [(data2, data3, data4, data5, data8, data9, data10, data11, data12, data6, data7)]
    return dict1


def main(seed):
    init_seeds(seed)

    """Load preprocessed data."""
    all_encode_prot5 = '/mydata/houdongliang/EnsemPPIS-master2/code/features/protein-DNA/129_siteT5.pkl'
    all_encode_prostt5 = '/mydata/houdongliang/EnsemPPIS-master2/code/features/protein-DNA/129_prostt5.pkl'
    fasta_file = '/mydata/houdongliang/EnsemPPIS-master2/code/dataset/protein-DNA/DNA-129_Test.fasta'
    output_file2 = '/mydata/houdongliang/EnsemPPIS-master2/code/dataset/protein-DNA/out' + '.txt'
    model_path="/mydata/houdongliang/EnsemPPIS-master2/code/model/protein-DNA.pkl"
    feature_ss_rsa='/mydata/houdongliang/EnsemPPIS-master2/code/features/protein-DNA/test_rsa_ss.csv'
    with open(feature_ss_rsa, newline='') as file12:
        dict11 = feature2(file12)
    with open(all_encode_prot5, "rb") as fp_enc1:
        all_encodes1 = pickle.load(fp_enc1)
    with open(all_encode_prostt5, "rb") as fp_enc2:
        all_encodes2 = pickle.load(fp_enc2)
    f222 = open(output_file2, 'a')


    """ create model and tester """

    protein_dim1 =247
    local_dim1 =247
    hid_dim = 64
    n_layers = 3
    n_heads = 8
    pf_dim = 256
    dropout = 0.1
    kernel_size = 7

    encoder1 = Encoder_cnn(protein_dim1, hid_dim, n_layers, kernel_size, dropout, device)
    encoder2 = Encoder_lstm(protein_dim1, hid_dim, n_layers, kernel_size, dropout, device)
    encoder3 = Encoder_att(local_dim1, hid_dim, n_layers, n_heads, pf_dim, DecoderLayer1, SelfAttention_no,
                           PositionwiseFeedforward1, dropout, device)
    decoder2 = Decoder(local_dim1, hid_dim, n_layers, n_heads, pf_dim, DecoderLayer2, SelfAttention,
                       PositionwiseFeedforward2, dropout, device)

    model2 = Predictor(encoder1, encoder2, encoder3, decoder2, device)

    model2.load_state_dict(torch.load(
        model_path,
        map_location=torch.device('cpu')))

    model2.to(device)

    tester2 = Predictor_test(model2)

    """Output files."""
    headers, sequences = read_fasta(fasta_file)
    for index, (header, seq) in enumerate(zip(headers, sequences), 1):
        name = header
        protein1 = all_encodes1.get(name, None)
        protein2 = all_encodes2.get(name, None)
        rsa = []
        data3 = []
        data4 = []
        data5 = []
        data6 = []
        data7 = []
        data8 = []
        data9 = []
        data10 = []
        data11 = []
        data12 = []
        print(name)
        all_2fetaure = dict11[name]
        for i in all_2fetaure:
            rsa.append(float(i[0]))
            data3.append(float(i[1]))
            data4.append(float(i[2]))
            data5.append(float(i[3]))
            data8.append(float(i[4]))
            data9.append(float(i[5]))
            data10.append(float(i[6]))
            data11.append(float(i[7]))
            data12.append(float(i[8]))
            data6.append(float(i[9]))
            data7.append(float(i[10]))

        aas = seq
        _prop = np.zeros((24 + 1 + 7 + 11, len(aas)))  # 32
        for i, a in enumerate(seq):
            aa = map_res[a]
            _prop[0:24, i] = blosummap[aanamemap[aa]]
            _prop[24, i] = min(i, len(aas) - i) * 1.0 / len(aas) * 2
            _prop[25:32, i] = meiler_features[aa] / 5
            _prop[32, i] = rsa[i]
            _prop[33, i] = data3[i]
            _prop[34, i] = data4[i]
            _prop[35, i] = data5[i]
            _prop[36, i] = data8[i]
            _prop[37, i] = data9[i]
            _prop[38, i] = data10[i]
            _prop[39, i] = data11[i]
            _prop[40, i] = data12[i]
            _prop[41, i] = data6[i]
            _prop[42, i] = data7[i]
        _prop_all = _prop.T
        _prop_all1 = np.expand_dims(_prop_all, axis=0)
        protein11= np.expand_dims(protein1, axis=0)
        protein22 = np.expand_dims(protein2, axis=0)
        print('1', protein11.shape,protein22.shape,_prop_all1.shape,name)

        test_loader2 = (protein11,protein22,  _prop_all1, index)
        predicted_labels_test2, predicted_scores_test2 = tester2.test(test_loader2, device)

        for i in range(len(predicted_labels_test2)):
            f222.write(name + '\t' + str(seq[i]) + '\t' + str(predicted_labels_test2[i]) + '\t' + str(
                predicted_scores_test2[i]) + '\n')

if __name__ == "__main__":
    """CPU or GPU"""
    # if torch.cuda.is_available() :
    #     os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    #     device = torch.device('cuda:0')
    #
    #     print('The code uses GPU...')
    # else:
    device = torch.device('cpu')
    print('The code uses CPU!!!')

    SEED = 1
    main(SEED)

