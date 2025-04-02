import torch
from transformers import AutoTokenizer
from models.mrscore import MRScore
from argparse import Namespace

# 设置模型和 LoRA 参数（与你训练时一致）
args = {
    "llm_model": "meta-llama/Llama-3.2-3B",
    "lora_inference": True,
    "llm_r": 16,
    "llm_alpha": 16,
    "lora_dropout": 0.1,
    "delta_file": "/content/R2GenGPT/save/v1/checkpoints/checkpoint_epoch27_step2548_val_loss1.792795.pth",
    "savedmodel_path": "./save/v1"
}

args = Namespace(**args)

# 初始化 MRScore 模型（加载 LoRA 并载入 delta 参数）
model = MRScore(args)
model.eval()

# 加载 tokenizer
tokenizer = AutoTokenizer.from_pretrained(args.llm_model, use_fast=True)
tokenizer.pad_token = tokenizer.eos_token


GT_answer = "The lungs are clear. No signs of pneumonia."
generated_answer = "There is no evidence of consolidation or pneumothorax."

# 构造输入格式
prompt = f"Ground Truth Answer:\n{GT_answer}\n\nGenerated Answer:\n{generated_answer}"
encoded = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=350)

input_ids = encoded["input_ids"]
attention_mask = encoded["attention_mask"]

# 推理打分
with torch.no_grad():
    output = model.model(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
    score = torch.sigmoid(output["logits"]).item()

print(f"✅ MRScore: {score:.4f}")
