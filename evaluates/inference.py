'''
Generate mrscore_prediction.csv to be evaluate.

use command:
!python ./evaluates/inference.py ./mrscore_evaluate_dataset.json \
  --delta_file /content/R2GenGPT/save/v1/checkpoints/checkpoint_epoch27_step2548_val_loss1.792795.pth \
  --output_file ./results/mrscore_output.csv \
  --lora_inference true

'''

import os
import json
import csv
import torch
import argparse
from argparse import Namespace
from transformers import AutoTokenizer
from models.mrscore import MRScore

def str2bool(v):
    # 支持 true/false 作为命令行参数
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "1", "True"):
        return True
    elif v.lower() in ("no", "false", "f", "0", "False"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")

def main(input_file, delta_file, output_file, lora_inference):
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
    model = MRScore(args)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(args.llm_model, use_fast=True)
    tokenizer.pad_token = tokenizer.eos_token

    # 读取数据集
    with open(input_file, "r") as f:
        dataset = json.load(f)

    # output name
    output_file = os.path.join("./results", output_file)

    # 写入 CSV 文件
    with open(output_file, mode="w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["gt_score", "marked_score"])

        for item in dataset:
            gt = item["gt_answer"]
            gen = item["generated_answer"]
            gt_score = item["score"]

            prompt = f"Ground Truth Answer:\n{gt}\n\nGenerated Answer:\n{gen}"
            encoded = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=350)

            with torch.no_grad():
                output = model.model(input_ids=encoded["input_ids"],
                                     attention_mask=encoded["attention_mask"],
                                     return_dict=True)
                marked_score = torch.sigmoid(output["logits"]).item() * 3

            writer.writerow([gt_score, round(marked_score, 6)])

    print(f"✅ 推理完成，结果已保存至 {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch inference with MRScore")
    parser.add_argument("input_file", type=str, help="Path to the input JSON file")
    parser.add_argument("--delta_file", type=str, required=True, help="Path to LoRA delta .pth file")
    parser.add_argument("--output_file", type=str, default="mrscore_predictions.csv", help="Path to output CSV file")
    parser.add_argument("--lora_inference", type=str2bool, default=True, help="Whether to use LoRA inference (True/False)")

    args = parser.parse_args()
    main(args.input_file, args.delta_file, args.output_file, args.lora_inference)

