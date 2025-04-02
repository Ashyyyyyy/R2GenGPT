'''
Generate mrscore_prediction.csv to be evaluate
'''

import json
import csv
import torch
import argparse
from argparse import Namespace
from transformers import AutoTokenizer
from models.mrscore import MRScore

def main(input_file):
    # 设置 LoRA 参数
    args = {
        "llm_model": "meta-llama/Llama-3.2-3B",
        "lora_inference": True,
        "llm_r": 32,
        "llm_alpha": 64,
        "lora_dropout": 0.1,
        "delta_file": "/content/R2GenGPT/save/v1/checkpoints/checkpoint_epoch27_step2548_val_loss1.792795.pth",
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

    # 设置输出文件名（自动命名）
    output_file = "mrscore_prediction.csv"

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
    args = parser.parse_args()
    main(args.input_file)
