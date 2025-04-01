import os
# from lightning.pytorch.loggers import CSVLogger
# from lightning.pytorch import loggers as pl_loggers
# from lightning.pytorch.callbacks import LearningRateMonitor
# from lightning.pytorch.callbacks import ModelCheckpoint

from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint


def add_callbacks(args):
    log_dir = args.savedmodel_path
    os.makedirs(log_dir, exist_ok=True)

    # --------- Add Callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(log_dir, "checkpoints"),
        filename="{epoch}",
        save_top_k=-1,
        save_last=True,
        save_weights_only=False,
        every_n_train_steps=args.every_n_train_steps,
        every_n_epochs=args.every_n_epochs,
        save_on_train_epoch_end=True
    )
    
    lr_monitor_callback = LearningRateMonitor(logging_interval='step')
    tb_logger = pl_loggers.TensorBoardLogger(save_dir=os.path.join(log_dir, "logs"), name="tensorboard")
    csv_logger = CSVLogger(save_dir=os.path.join(log_dir, "logs"), name="csvlog")

    to_returns = {
        "callbacks": [checkpoint_callback, lr_monitor_callback],
        "loggers": [csv_logger, tb_logger]
    }
    return to_returns
