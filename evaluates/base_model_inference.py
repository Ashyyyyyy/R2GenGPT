# llama3_prompt_scoring.py

import json
import csv
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm

# MODEL_NAME = "meta-llama/Llama-3.2-3B"
# MODEL_NAME = "ministral/Ministral-3b-instruct"
# MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.2"
MODEL_NAME = "meta-llama/Llama-2-7b-hf"
# MODEL_NAME = "meta-llama/Llama-2-7b-chat-hf"


output_file = "Llama-2-7b-hf_prompt_score_output.csv"

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


    # for large model over 3b
    prompt = f"""You are a medical expert evaluating generated radiology answers.

    Ground Truth Answer:
    {gt_answer}

    Generated Answer:
    {gen_answer}

    Please rate the quality of the generated answer from 0 (worst) to 3 (best), awarding 1 point for each of the following criteria:

    1. Grammar: The answer is well-formed and grammatically correct.
    2. Information Retrieval: The answer is relevant to the image-question pair and demonstrates appropriate knowledge and information retrieval.
    3. Medical Assessment: The answer addresses the clinical/medical problem appropriately.

    ONLY respond with a single number between 0 and 3. Do NOT include any explanation or extra text.

    Total Score (0–3):
    """

    # # for small model under 3b
    # prompt = f"""You are a medical expert evaluating generated radiology answers.

    # Ground Truth Answers:
    # {gt_answer}

    # Generated Answers:
    # {gen_answer}

    # Please rate the quality of the generated answers from 0 (worst) to 3 (best), based on grammar, information retrieval, and medical aassessment.

    # Score:"""

    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=3,
            do_sample=False,
            num_beams=1,
            temperature=0.0,
            early_stopping=True,
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

    results.append({"gt_score": gt_score, "marked_score": score, "model_output": generated_score_text})

# 写入 CSV
with open(output_file, "w", newline='') as f:
    writer = csv.DictWriter(f, fieldnames=["gt_score", "marked_score", "model_output"])
    writer.writeheader()
    writer.writerows(results)

print(f"✅ Finished scoring with {MODEL_NAME} method.")