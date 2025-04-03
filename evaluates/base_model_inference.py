# llama3_prompt_scoring.py

import json
import csv
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm

MODEL_NAME = "meta-llama/Llama-3.2-3B"  # 也可以用你自己的权重路径

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 加载 tokenizer 和 causal 模型（非分类模型）
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
model.to(device)
model.eval()

# 加载数据
with open("./data/mrscore_evaluate_dataset.json", "r") as f:
    dataset = json.load(f)

results = []

for entry in tqdm(dataset):
    gt_score = entry["score"]
    gt_answer = entry["gt_answer"]
    gen_answer = entry["generated_answer"]

    prompt = f"""You are a medical expert evaluating generated radiology reports.

    Ground Truth Report:
    {gt_answer}

    Generated Report:
    {gen_answer}

    Please rate the quality of the generated report from 0 (worst) to 3 (best), awarding 1 point for each of the following criteria:

    1. Grammar: The answer is well-formed and grammatically correct.
    2. Information Retrieval: The answer is relevant to the image-question pair and demonstrates appropriate knowledge and information retrieval.
    3. Medical Assessment: The answer addresses the clinical/medical problem appropriately.

    Total Score (0–3):
    """

    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=10,
            do_sample=False,
            num_beams=1,
            temperature=0.0,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    # 解码并提取数字
    decoded = tokenizer.decode(output[0], skip_special_tokens=True)
    generated_score_text = decoded[len(prompt):].strip()

    try:
        score = float(generated_score_text.split()[0])
        score = round(max(0, min(3, score)), 2)
    except Exception:
        score = -1  # 无法识别

    results.append({"gt_score": gt_score, "marked_score": score})

# 写入 CSV
with open("llama3_prompt_score_output.csv", "w", newline='') as f:
    writer = csv.DictWriter(f, fieldnames=["gt_score", "marked_score"])
    writer.writeheader()
    writer.writerows(results)

print("✅ Finished scoring with LLaMA-3 prompt method.")