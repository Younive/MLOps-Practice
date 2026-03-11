import torch
import wandb
import pandas as pd
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from data import DataModule
from model import ColaModel

import hydra
from omegaconf.omegaconf import OmegaConf
import logging

logger = logging.getLogger(__name__)

class SimpleVisualizationLogger(pl.Callback):
    def __init__(self, datamodule):
        super().__init__()
        self.datamodule = datamodule

    def on_validation_end(self, trainer, pl_module):
        val_bacth = next(iter(self.datamodule.val_dataloader()))
        sentences = val_bacth['sentence']
        labels = val_bacth['label']
        outputs = pl_module(val_bacth['input_ids'], val_bacth['attention_mask'])
        pred = torch.argmax(outputs.logits, dim=1)

        df = pd.DataFrame({
            'sentence': sentences,
            'label': labels.cpu().numpy(),
            'pred': pred.cpu().numpy()
        })
        wrong_predictions_df = df[df['label'] != df['pred']]
        trainer.logger.experiment.log(
            {
                "wrong_predictions": wandb.Table(dataframe=wrong_predictions_df, allow_mixed_types=True),
                "global_step": trainer.global_step
            }
        )

@hydra.main(version_base=None, config_path="config", config_name="config")
def main(cfg):
    logging.info(OmegaConf.to_yaml(cfg, resolve=True))
    logger.info(f"using model:{cfg.model.name}")
    logger.info(f"using tokenizer:{cfg.model.tokenizer}")

    cola_data = DataModule(
        model_name=cfg.model.tokenizer,
        batch_size=cfg.processing.batch_size,
    )

    cola_model = ColaModel(
        model_name=cfg.model.name,
    )

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
    wandb_logger = WandbLogger(project="cola_model", name="cola_run_1")
    trainer = pl.Trainer(
        default_root_dir = 'logs',
        gpus = (1 if torch.cuda.is_available() else 0),
        max_epochs = 1,
        log_every_n_steps = cfg.training.log_every_n_steps,
        limit_train_batches = cfg.training.limit_train_batches,
        limit_val_batches = cfg.training.limit_val_batches,
        deterministic = cfg.training.deterministic,
        fast_dev_run = True,
        logger = wandb_logger,
        callbacks = [checkpoint_callback, SimpleVisualizationLogger(cola_data), early_stopping_callback]
    )
    trainer.fit(cola_model, cola_data)
    wandb.finish()

if __name__ == "__main__":
    main()