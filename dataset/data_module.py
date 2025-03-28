# data_module.py

import sys
# sys.path.append('/apdcephfs/share_733425/vinnylywang/zhanyuwang/Code/xray_chat')
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader
from dataset.data_helper import create_datasets
import torch
import numpy as np
from torch.nn.utils.rnn import pad_sequence


def collate_func(batch):
    elem = batch[0]
    res = {}
    for key, v in elem.items():
        value = [d[key] for d in batch]
        if isinstance(v, str):
             res[key] = value
        elif isinstance(v, torch.Tensor):
             if 'input_ids' in key:
                value = pad_sequence(value, batch_first=True)
             else:
                value = torch.stack(value, 0)
             res[key] = value
        elif isinstance(v, np.ndarray):
             value = torch.tensor(np.stack(value))
             res[key] = value
        elif isinstance(v, int):
             res[key] = torch.tensor(value)
        else:
             print(key)
             print('unkown data type')
    return res


def custom_collate_fn(batch):
    ids = [item["id"] for item in batch]
    texts = [item["text"] for item in batch]
    if 'image' in batch[0]:
        images = [item["image"] for item in batch]
        return {"ids": ids, "texts": texts, "images": images}
    else:
        return {"ids": ids, "texts": texts}


class DataModule(LightningDataModule):

    def __init__(
            self,
            args
    ):
        super().__init__()
        self.args = args

    def prepare_data(self):
        """
        Use this method to do things that might write to disk or that need to be done only from a single process in distributed settings.

        download

        tokenize

        etc…
        :return:
        """

    def setup(self, stage: str):
        """
        There are also data operations you might want to perform on every GPU. Use setup to do things like:

        count number of classes

        build vocabulary

        perform train/val/test splits

        apply transforms (defined explicitly in your datamodule or assigned in init)

        etc…
        :param stage:
        :return:
        """
        train_dataset, dev_dataset, test_dataset = create_datasets(self.args)
        self.dataset = {
            "train": train_dataset, "validation": dev_dataset, "test": test_dataset
        }
        # print("[DEBUG] Validation set size:", len(dev_dataset))

    def train_dataloader(self):
        """
        Use this method to generate the train dataloader. Usually you just wrap the dataset you defined in setup.
        :return:
        """
        loader = DataLoader(self.dataset["train"], batch_size=self.args.batch_size, drop_last=True, pin_memory=True,
                        num_workers=self.args.num_workers, prefetch_factor=self.args.prefetch_factor)
        return loader


    def val_dataloader(self):
        """
        Use this method to generate the val dataloader. Usually you just wrap the dataset you defined in setup.
        :return:
        """
        loader = DataLoader(self.dataset["validation"], batch_size=self.args.val_batch_size, drop_last=False, pin_memory=True,
                            num_workers=self.args.num_workers, prefetch_factor=self.args.prefetch_factor)
        return loader


    def test_dataloader(self):
        loader = DataLoader(self.dataset["test"], batch_size=self.args.val_batch_size, drop_last=False, pin_memory=False,
                        num_workers=self.args.num_workers, prefetch_factor=self.args.prefetch_factor)
        return loader
    

if __name__ == '__main__':
    from configs.config import parser
    args = parser.parse_args()
    loader = DataModule(args)
    loader.setup(stage=None)
    train = loader.train_dataloader()

    for data in train:
        print(data)