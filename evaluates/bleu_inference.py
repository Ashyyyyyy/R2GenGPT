import json
import os
from evalcap.bleu.bleu import Bleu

# 读取你的 json 数据
with open(r"H:\Research\QAGeneration\mrscore_evaluate_dataset.json", "r") as f:
    data = json.load(f)

# 构造 BLEU 输入格式
gts = {}
res = {}

for i, item in enumerate(data):
    key = f"sample_{i}"
    gts[key] = [item["gt_answer"].strip().lower()]  # 参考答案
    res[key] = [item["generated_answer"].strip().lower()]  # 生成答案

# 计算 BLEU 分数
bleu = Bleu(n=4)
score, scores = bleu.compute_score(gts, res, verbose=1)

print("== Overall BLEU scores ==")
for i in range(4):
    print(f"BLEU-{i+1}: {score[i]:.4f}")
