#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
from collections import defaultdict
from sklearn.metrics import (
    confusion_matrix,
    roc_auc_score,
    precision_recall_curve,
    auc,
    accuracy_score,
    matthews_corrcoef
)

prediction_txt = ""
test_csv = ""

# ==========================
# 读取真实标签
# ==========================
df = pd.read_csv(test_csv)

label_dict = {}

for _, row in df.iterrows():

    name = str(row.iloc[0]).strip()

    label_str = str(row.iloc[2]).strip()

    label_dict[name] = [int(x) for x in label_str]

# ==========================
# 读取预测结果
# ==========================
pred_label_dict = defaultdict(list)
pred_score_dict = defaultdict(list)

with open(prediction_txt) as f:

    for line in f:

        cols = line.strip().split()

        if len(cols) < 4:
            continue

        name = cols[0]

        pred_label = int(cols[2])

        pred_score = float(cols[3])

        pred_label_dict[name].append(pred_label)
        pred_score_dict[name].append(pred_score)

# ==========================
# 汇总所有蛋白
# ==========================
all_true = []
all_pred = []
all_score = []

for name in pred_label_dict:

    if name not in label_dict:
        print(f"Warning: {name} not found")
        continue

    y_true = label_dict[name]
    y_pred = pred_label_dict[name]
    y_score = pred_score_dict[name]

    if len(y_true) != len(y_pred):
        print(
            f"Length mismatch: "
            f"{name} true={len(y_true)} pred={len(y_pred)}"
        )
        continue

    all_true.extend(y_true)
    all_pred.extend(y_pred)
    all_score.extend(y_score)

print("Total residues:", len(all_true))

# ==========================
# 计算指标
# ==========================
ACC = accuracy_score(all_true, all_pred)

AUC = roc_auc_score(all_true, all_score)

TN, FP, FN, TP = confusion_matrix(
    all_true,
    all_pred,
    labels=[0, 1]
).ravel()

Recall = TP / (TP + FN + 1e-8)

Precision = TP / (TP + FP + 1e-8)

F1 = 2 * Precision * Recall / (
    Precision + Recall + 1e-8
)

MCC = matthews_corrcoef(
    all_true,
    all_pred
)

precision_curve, recall_curve, _ = precision_recall_curve(
    all_true,
    all_score
)

PRC = auc(recall_curve, precision_curve)

Specificity = TN / (TN + FP + 1e-8)

# ==========================
# 输出
# ==========================
print("\n===== Overall Metrics =====")

print(f"ACC         : {ACC:.4f}")
print(f"AUC         : {AUC:.4f}")
print(f"Recall      : {Recall:.4f}")
print(f"Precision   : {Precision:.4f}")
print(f"F1          : {F1:.4f}")
print(f"MCC         : {MCC:.4f}")
print(f"PRC(AUPRC)  : {PRC:.4f}")
print(f"Specificity : {Specificity:.4f}")