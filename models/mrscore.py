import os
import torch
import torch.nn.functional as F
from typing import List
# import lightning.pytorch as pl
import pytorch_lightning as pl
import einops
from functools import partial
from transformers import AutoTokenizer
from torchmetrics.text import BLEUScore
import numpy as np
from deepspeed.ops.adam import DeepSpeedCPUAdam
from torch.optim.lr_scheduler import StepLR
from peft import get_peft_model, LoraConfig, TaskType 
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from PIL import Image


class MRScore(pl.LightningModule):
    """
    MRScoreModel.
    """
    def __init__(self, args):
        super().__init__()

        if isinstance(args, dict): # if load from ckpt, we need to change the types
            from argparse import Namespace
            args = Namespace(**args)

        self.args = args
        self.save_hyperparameters(args)

        print('Loading LLM Model')
        self.tokenizer = AutoTokenizer.from_pretrained(self.hparams.llm_model, use_fast=True)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForSequenceClassification.from_pretrained(self.hparams.llm_model, num_labels=1, torch_dtype=torch.bfloat16)
        self.model.config.pad_token_id = self.tokenizer.pad_token_id
        peft_config = LoraConfig(
            task_type=TaskType.SEQ_CLS,
            inference_mode=self.hparams.lora_inference,
            r=self.hparams.llm_r,
            lora_alpha=self.hparams.llm_alpha,
            lora_dropout=self.hparams.lora_dropout,
            )
        
        self.model = get_peft_model(self.model, peft_config)
        self.model.print_trainable_parameters()
        print('Loading LLM-LoRA Done')
        
        self.val_step_outputs = []
        self.val_score = 100.0

        if args.delta_file is not None:
            state_dict = torch.load(args.delta_file, map_location='cpu', weights_only=False)['model']
            self.load_state_dict(state_dict=state_dict, strict=False)
            print(f'Load checkpoint from {args.delta_file}')


    def custom_loss(self, rewards_chosen, rewards_rejected, margin, alpha=0.5):
        hinge_loss = F.relu(margin - (rewards_chosen - rewards_rejected))
        squared_difference_loss = (rewards_chosen - rewards_rejected - margin)**2
        combined_loss = alpha * hinge_loss + (1 - alpha) * squared_difference_loss
        return combined_loss.mean()


    def compute_loss(self, inputs, return_outputs=True):
        rewards_chosen = F.sigmoid(self.model(
            input_ids=inputs["input_ids_chosen"],
            attention_mask=inputs["attention_mask_chosen"],
            return_dict=True,
            )["logits"])
        rewards_rejected = F.sigmoid(self.model(
            input_ids=inputs["input_ids_rejected"],
            attention_mask=inputs["attention_mask_rejected"],
            return_dict=True,
            )["logits"])
        # calculate loss, optionally modulate with margin
        # if "margin" in inputs:
        #     loss = -F.logsigmoid(rewards_chosen - rewards_rejected - inputs["margin"]).mean()
        # else:
        #     loss = -F.logsigmoid(rewards_chosen - rewards_rejected).mean()
            
        loss = self.custom_loss(rewards_chosen, rewards_rejected, inputs["margin"])
        
        if return_outputs:
                return loss, {
                    "rewards_chosen": rewards_chosen,
                    "rewards_rejected": rewards_rejected,
                }
        return loss

    def training_step(self, batch, batch_idx):
        to_log = {}
        loss, logits = self.compute_loss(inputs=batch)      
        to_log['loss'] = loss
        self.log_dict(to_log, prog_bar=True)
        return loss

    def save_checkpoint(self, eval_res):
        current_epoch, global_step = self.trainer.current_epoch, self.trainer.global_step
        param_grad_dic = {
            k: v.requires_grad for (k, v) in self.named_parameters() if v.requires_grad
        }
        state_dict = self.state_dict()
        for k in list(state_dict.keys()):
            if k not in param_grad_dic.keys():
                del state_dict[k]
        save_obj = {
            "model": state_dict,
            "config": self.hparams,
            "epoch": current_epoch,
            "step":global_step
        }
        os.makedirs(os.path.join(self.hparams.savedmodel_path, 'checkpoints'), exist_ok=True)
        save_to = os.path.join(
            self.hparams.savedmodel_path, 'checkpoints',
            "checkpoint_epoch{}_step{}_val_loss{:3f}.pth".format(current_epoch, global_step, eval_res),
        )
        self.print("Saving checkpoint at step {} to {}.".format(global_step, save_to))
        torch.save(save_obj, save_to)
    
    def validation_step(self, batch, batch_idx):
        to_log = {}
        loss, logits = self.compute_loss(inputs=batch)      
        to_log['val_loss'] = loss
        self.val_step_outputs.append({"val_loss": loss})
        return to_log

    def on_validation_epoch_end(self):
        val_loss = []
        print("[DEBUG] Validation epoch end triggered ✅")
        for i in self.val_step_outputs:
            val_loss.append(i['val_loss'].item())
        val_loss = np.mean(val_loss)
        if self.trainer.local_rank == 0:
            print("[DEBUG] Save  checkpoint triggered ✅")
            self.save_checkpoint(val_loss)

    def configure_optimizers(self):
        if 'deepspeed' in self.hparams.strategy:
            optimizer = DeepSpeedCPUAdam(self.parameters(), lr=self.hparams.learning_rate)
            scheduler = StepLR(optimizer, step_size=1, gamma=0.85)
            return {"optimizer": optimizer, "lr_scheduler": scheduler}
        else:
            optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
            scheduler = StepLR(optimizer, step_size=1, gamma=0.85)
            return {"optimizer": optimizer, "lr_scheduler": scheduler}

    def get_progress_bar_dict(self):
        # don't show the version number
        items = super().get_progress_bar_dict()
        items.pop("v_num", None)
        return items

    def optimizer_zero_grad(self, epoch, batch_idx, optimizer):
        optimizer.zero_grad()