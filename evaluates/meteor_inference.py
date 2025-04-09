import json
import pandas as pd
from nltk.translate.meteor_score import meteor_score
from statistics import mean

# Load the dataset
file_path = r"H:\Research\QAGeneration\mrscore_evaluate_dataset.json"
with open(file_path, "r", encoding="utf-8") as f:
    data = json.load(f)


# Compute METEOR scores
marked_scores = []
human_scores = []
for i, item in enumerate(data):
    reference = item["gt_answer"].split()
    hypothesis = item["generated_answer"].split()
    human_scores.append(item["score"])
    score = meteor_score([reference], hypothesis)
    marked_scores.append(score)

df = pd.DataFrame({
    "gt_score": human_scores,
    "marked_score": marked_scores,
})
csv_path = r"H:\Research\results\meteor_score_output.csv"
df.to_csv(csv_path, index=False)
print("Saved to:", csv_path)