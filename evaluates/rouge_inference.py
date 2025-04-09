import evaluate
import json
import pandas as pd

# 载入数据
with open(r"H:\Research\QAGeneration\mrscore_evaluate_dataset.json", "r") as f:
    data = json.load(f)

# 提取参考答案和生成答案
references = [entry["gt_answer"] for entry in data]
predictions = [entry["generated_answer"] for entry in data]
human_scores = [entry["score"] for entry in data]

# 加载 ROUGE 评估工具
rouge = evaluate.load("rouge")

# 计算 ROUGE 分数
results = rouge.compute(predictions=predictions, references=references, use_aggregator=False)

# 提取每条 ROUGE-L f1 分数
rouge_l_f1_list = results["rougeL"]

df = pd.DataFrame({
    "gt_score": human_scores,
    "marked_score": rouge_l_f1_list,
})
csv_path = r"H:\Research\results\rouge_score_output.csv"
df.to_csv(csv_path, index=False)

