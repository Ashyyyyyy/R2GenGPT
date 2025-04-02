'''
Generate mrscore_prediction.csv to be evaluated.

Example:
!python ./evaluates/inference.py ./mrscore_evaluate_dataset.json \
  --delta_file /content/R2GenGPT/save/v1/checkpoints/checkpoint_epoch27_step2548_val_loss1.792795.pth \
  --output_file ./results/mrscore_output.csv \
  --lora_inference true \
  --batch_size 16
'''

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import json
import csv
import torch
import argparse
from argparse import Namespace
from transformers import AutoTokenizer
from models.mrscore import MRScore
from tqdm import tqdm

def str2bool(v):
    if isinstance(v, bool): return v
    if v.lower() in ("yes", "true", "t", "1", "True"): return True
    elif v.lower() in ("no", "false", "f", "0", "False"): return False
    else: raise argparse.ArgumentTypeError("Boolean value expected.")

def main(input_file, delta_file, output_file, lora_inference, batch_size):
    # Device

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"💻 Using device: {device}")

    # 设置 LoRA 参数
    args = {
        "llm_model": "meta-llama/Llama-3.2-3B",
        "lora_inference": lora_inference,
        "llm_r": 32,
        "llm_alpha": 64,
        "lora_dropout": 0.1,
        "delta_file": delta_file,
        "savedmodel_path": "./save/v1"
    }
    args = Namespace(**args)

    # 初始化模型和 tokenizer
    print("🚀 Loading MRScore model with LoRA...")
    model = MRScore(args)
    model = model.to(device)
    model.eval()
    print("✅ Model loaded.")

    print("🔤 Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.llm_model, use_fast=True)
    tokenizer.pad_token = tokenizer.eos_token
    print("✅ Tokenizer loaded.")

    # 加载数据
    with open(input_file, "r") as f:
        dataset = json.load(f)

    os.makedirs(os.path.dirname("./results"), exist_ok=True)
    output_file = os.path.join("./results", output_file)

    # 写入 CSV 文件
    with open(output_file, mode="w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["gt_score", "marked_score"])

        # 批量推理
        for i in tqdm(range(0, len(dataset), batch_size), desc="Inferencing", ncols=80):
            batch = dataset[i:i + batch_size]

            prompts = [
                f"Ground Truth Answer: {item['gt_answer']} \n\n Generated Answer: {item['generated_answer']}"
                for item in batch
            ]
            gt_scores = [item["score"] for item in batch]

            encoded = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=350)
            input_ids = encoded["input_ids"].to(device)
            attention_mask = encoded["attention_mask"].to(device)

            with torch.no_grad():
                output = model.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    return_dict=True
                )
                scores = torch.sigmoid(output["logits"]).squeeze(-1) * 3  # [batch_size]

            for gt_score, marked_score in zip(gt_scores, scores):
                writer.writerow([gt_score, round(marked_score.item(), 6)])

    print(f"✅ 推理完成！结果保存在: {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch inference with MRScore")
    parser.add_argument("input_file", type=str, help="Path to the input JSON file")
    parser.add_argument("--delta_file", type=str, required=True, help="Path to LoRA delta .pth file")
    parser.add_argument("--output_file", type=str, default="mrscore_predictions.csv", help="Path to output CSV file")
    parser.add_argument("--lora_inference", type=str2bool, default=True, help="Whether to use LoRA inference (True/False)")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for inference")

    args = parser.parse_args()
    main(args.input_file, args.delta_file, args.output_file, args.lora_inference, args.batch_size)
