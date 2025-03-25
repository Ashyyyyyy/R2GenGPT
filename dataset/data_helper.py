import sys
# sys.path.append('/apdcephfs/share_733425/vinnylywang/zhanyuwang/Code/xray_chat/')
import os
import json
import numpy as np
from numpy.random import randint
import random
from PIL import Image
import torch.utils.data as data
import torchvision.transforms as transforms
from transformers import AutoTokenizer
from transformers import LlamaTokenizer

class FieldParser:
    def __init__(
            self,
            args
    ):
        super().__init__()
        self.args = args
        self.tokenizer = AutoTokenizer.from_pretrained(self.args.llm_model, use_fast=True)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
    def tokenize(self, text):
        out = self.tokenizer(
            text,
            return_tensors="pt",
            padding='max_length',
            add_special_tokens=True,
            truncation=True,
            max_length=self.args.max_length)
        input_ids = out.input_ids[0]
        attention_mask = out.attention_mask[0]
        return input_ids, attention_mask

    def parse(self, features):
        chose = features['chosen']
        rejected = features['rejected']
        
        input_ids_chosen, attention_mask_chosen = self.tokenize(chose)
        input_ids_rejected, attention_mask_rejected = self.tokenize(rejected)
        margin = features['margin']
        to_return = {
            "input_ids_chosen": input_ids_chosen,
            "attention_mask_chosen": attention_mask_chosen,
            "input_ids_rejected": input_ids_rejected,
            "attention_mask_rejected": attention_mask_rejected,
            "margin": margin
        }
        return to_return

    def transform_with_parse(self, inputs):
        return self.parse(inputs)

class ParseDataset(data.Dataset):
    def __init__(self, args, split='train'):
        self.train = split == "train"
        meta = json.load(open(args.dataset, 'r'))
        if split == "train":
            self.df = meta['train']
            # self.df = meta[:8000]
        else:
            self.df = meta['test']
        self.parser = FieldParser(args)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        try:
            return self.parser.transform_with_parse(self.df[index])
        except Exception as e:
            print(e)

def create_datasets(args):
    train_dataset = ParseDataset(args, 'train')
    dev_dataset = ParseDataset(args, 'val')
    test_dataset = ParseDataset(args, 'test')
    return train_dataset, dev_dataset, test_dataset


if __name__ == '__main__':
    from tqdm import tqdm
    from configs.config import parser
    args = parser.parse_args()
    loader = ParseDataset(args)

    for i in tqdm(range(loader.__len__())):
        data = loader.__getitem__(i)

