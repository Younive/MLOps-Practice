import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from data import DataModule
from model import ColaModel


cola_data = DataModule()
cola_model = ColaModel()
wandb_logger = WandbLogger(project="cola_model", name="cola_run_1")

def main():
    checkpoint_callback = ModelCheckpoint(
    dirpath="./models",
    monitor="val_acc",
    mode="min",
    )
    early_stopping_callback = EarlyStopping(
        monitor="val_acc",
        patience=3,
        verbose=True,
        mode="min",
    )
    trainer = pl.Trainer(
        default_root_dir = 'logs',
        gpus = (1 if torch.cuda.is_available() else 0),
        max_epochs = 3,
        fast_dev_run = True,
        logger = wandb_logger,
        callbacks = [checkpoint_callback, early_stopping_callback]
    )
    trainer.fit(cola_model, cola_data)

if __name__ == "__main__":
    main()