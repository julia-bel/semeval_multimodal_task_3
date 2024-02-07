import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "6"

import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader, ConcatDataset

from semeval.experiments.belikova.videollama.models import (
    EmbeddedCausalClassifier,
)
from semeval.experiments.belikova.videollama.data import (
    EmbeddedDataset,
)


if __name__ == "__main__":
    # data preprocessing
    device = torch.device("cuda")
    modalities = ["text", "video"]
    train_dataset = EmbeddedDataset(
        split="train", root="/semeval/data/prompt_data/train"
    )
    val_dataset = EmbeddedDataset(split="val", root="/semeval/data/prompt_data/val")

    # cat_dataset = train_dataset
    # cat_dataset = ConcatDataset([train_dataset, val_dataset])

    train_loader = DataLoader(
        train_dataset,
        batch_size=1,
        shuffle=True,
        num_workers=4,
        collate_fn=val_dataset.collater,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        num_workers=4,
        collate_fn=val_dataset.collater,
    )

    # model settings
    model = EmbeddedCausalClassifier(modalities)

    # train params
    max_epochs = 100
    output_path = "/semeval/models/causal_best"

    # training
    wandb_logger = WandbLogger(name="causal_best", project="emotion_analysis")

    checkpoint_callback = ModelCheckpoint(
        dirpath=output_path,
        filename="{epoch}",
        save_top_k=-1,
        every_n_epochs=5,
    )

    trainer = pl.Trainer(
        callbacks=[checkpoint_callback],
        max_epochs=max_epochs,
        accelerator="gpu",
        accumulate_grad_batches=6,
        gradient_clip_val=0.5,
        gradient_clip_algorithm="value",
        devices=-1,
        logger=wandb_logger,
    )
    trainer.fit(model, train_loader, val_loader)

    wandb_logger.experiment.save(output_path)
    wandb_logger.experiment.finish()
