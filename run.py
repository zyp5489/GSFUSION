import os
import pytorch_lightning as pl
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.callbacks import ModelCheckpoint

from config import param_dict
from src.lightning import testModel
from src.data.datamodule import testFpDataModule


def main():
    os.makedirs("lightning_logs/checkpoints", exist_ok=True)

    logger = CSVLogger(save_dir='lightning_logs/', name='experiment')
    checkpoint_callback = ModelCheckpoint(
        monitor='valid_mse',
        dirpath=os.path.join(os.getcwd(), 'lightning_logs/checkpoints'),
        filename='-{epoch:03d}-{valid_mse:.4f}--{valid_ci:.4f}',
        save_top_k=1,
        mode='min',
        save_last=True,
    )

    model = testModel(**param_dict)
    dm    = testFpDataModule(**param_dict)

    trainer = pl.Trainer(
        accelerator="gpu", devices=[0],
        max_epochs=2000,
        check_val_every_n_epoch=1,
        logger=logger,
        enable_progress_bar=True,
        callbacks=[checkpoint_callback],
    )

    trainer.fit(model=model, datamodule=dm)


if __name__ == "__main__":
    main()