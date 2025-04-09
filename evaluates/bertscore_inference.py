import json
import pandas as pd
from bert_score import score


# Load the dataset
file_path = r"H:\Research\QAGeneration\mrscore_evaluate_dataset.json"
with open(file_path, "r", encoding="utf-8") as f:
    data = json.load(f)

# 提取 gt_answer 和 generated_answer
refs = [item["gt_answer"] for item in data]
cands = [item["generated_answer"] for item in data]
human_scores = [item["score"] for item in data]

# 计算 BERTScore
P, R, F1 = score(cands, refs, lang="en")


df = pd.DataFrame({
    "gt_score": human_scores,
    "marked_score": F1,
})
csv_path = r"H:\Research\results\bertscore_score_output.csv"
df.to_csv(csv_path, index=False)
print("Saved to:", csv_path)
