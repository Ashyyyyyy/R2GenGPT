import json
import matplotlib.pyplot as plt
import pandas as pd
from evalcap.bleu.bleu import Bleu

# 加载数据
with open(r"H:\Research\QAGeneration\mrscore_evaluate_dataset.json", "r") as f:
    data = json.load(f)

# 准备 BLEU 输入格式
gts, res, human_scores = {}, {}, []

for i, item in enumerate(data):
    key = f"sample_{i}"
    gts[key] = [item["gt_answer"].strip().lower()]
    res[key] = [item["generated_answer"].strip().lower()]
    human_scores.append(item["score"])

# 计算 BLEU
bleu = Bleu(n=4)
_, scores = bleu.compute_score(gts, res)
bleu4_scores = [s[1] for s in zip(*scores)]


# 储存 csv
df = pd.DataFrame({
    "gt_score": human_scores,
    "marked_score": bleu4_scores,
})
csv_path = r"H:\Research\results\bleu4_score_output.csv"
df.to_csv(csv_path, index=False)

# 绘制散点图
plt.figure(figsize=(8, 6))
plt.scatter(bleu4_scores, human_scores, alpha=0.6)
plt.xlabel("BLEU-4 Score")
plt.ylabel("Human Score (0–3)")
plt.title("BLEU-4 Score vs Human Evaluation")
plt.grid(True)
plt.tight_layout()
plt.savefig(r"H:\Research\results\bleu4_score_output.png", dpi=300)
plt.show()
