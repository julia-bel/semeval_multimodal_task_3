import os

os.environ["WANDB_PROJECT"] = "semeval_emotion_classification"
# os.environ["WANDB_PROJECT"] = "semeval_cause_classification"
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

from semeval.experiments.kosenko.language_bind.languagebind_classification_video_text import (
    exp_1_load_dataloader,
    exp_1_load_dataset,
    exp_9_load_dataloader,
    exp_9_load_dataset,
)


import torch
from transformers.modeling_outputs import TokenClassifierOutput

from semeval.experiments.kosenko.language_bind.LanguageBind.languagebind import (
    LanguageBind,
    to_device,
    transform_dict,
    LanguageBindImageTokenizer,
)
from typing import Dict, List, Optional
from torch import nn
from torch.utils.data import Dataset
from transformers import Trainer
from transformers import TrainingArguments
import datasets

import numpy as np
import random
from peft import (
    get_peft_config,
    PeftModel,
    PeftConfig,
    get_peft_model,
    LoraConfig,
    TaskType,
)


def random_seed(seed=42, rank=0):
    torch.manual_seed(seed + rank)
    np.random.seed(seed + rank)
    random.seed(seed + rank)


random_seed()

from datasets import load_dataset
from torchvision.io import read_video
import json
import torch
import os
from torch.utils.data import Dataset, DataLoader
import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from transformers import MPNetForSequenceClassification, AutoTokenizer, AutoConfig


class CustomTrainer(Trainer):
    def compute_loss(
        self,
        model,
        inputs,
        return_outputs=False,
    ):
        custom_inputs = tokenizer(
            inputs["text"],
            # max_length=77,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        ).to(device)

        # forward pass
        outputs = model(**custom_inputs)
        label = inputs["label"].to(device)
        loss_func = torch.nn.CrossEntropyLoss()
        loss = loss_func(outputs.logits, label)
        return (loss, outputs) if return_outputs else loss

    def get_train_dataloader(self):
        train_dataset = self.train_dataset
        return DataLoader(
            train_dataset,
            batch_size=self.args.per_device_train_batch_size,
            shuffle=True,
        )

    def get_eval_dataloader(self, eval_dataset):
        return DataLoader(
            self.eval_dataset,
            batch_size=self.args.per_device_eval_batch_size,
            shuffle=False,
        )

    def prediction_step(
        self,
        model,
        inputs,
        prediction_loss_only,
        ignore_keys=None,
    ):
        model.eval()
        with torch.no_grad():
            custom_inputs = tokenizer(
                inputs["text"],
                # max_length=77,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            ).to(device)

            # forward pass
            outputs = model(**custom_inputs)
            outputs = outputs.logits
            label = inputs["label"].to(device)
            loss_func = torch.nn.CrossEntropyLoss()
            loss = loss_func(outputs, label)
        if prediction_loss_only:
            return (loss, None, None)

        return (loss, outputs, label)


def compute_metrics(eval_preds):
    predictions = eval_preds.predictions.argmax(-1)
    labels = eval_preds.label_ids

    f1_score_result = f1_score(
        labels,
        predictions,
        average="macro",
    )
    return {
        "f1_score": f1_score_result,
    }


def exp_12_load_model(labels=None):
    model_name = "sentence-transformers/all-mpnet-base-v2"
    config = AutoConfig.from_pretrained(model_name)
    config.num_labels = labels
    model = MPNetForSequenceClassification.from_pretrained(model_name, config=config)
    # for param in model.named_parameters():
    #     if "mpnet" in param[0]:
    #         param[1].requires_grad_(False)
    peft_config = LoraConfig(
        inference_mode=False,
        r=16,
        lora_alpha=16,
        lora_dropout=0.1,
        bias="all",
        target_modules=[
            "k",
            "v",
            "q",
            "o",
            "dense",
        ],
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    model.config = None
    return model


if __name__ == "__main__":
    all_emotions = [
        "surprise",
        "fear",
        "sadness",
        "neutral",
        "joy",
        "anger",
        "disgust",
    ]

    emotions2labels = {em: i for i, em in enumerate(all_emotions)}

    labels2emotions = {i: em for i, em in enumerate(all_emotions)}

    print(emotions2labels)

    print(labels2emotions)

    dataset = exp_1_load_dataset()
    training_data_list, test_data_list = dataset["train"], dataset["test"]
    # training_data_list = training_data_list.to_list()[:2]
    # test_data_list = test_data_list.to_list()[:2]
    training_data = exp_1_load_dataloader(training_data_list, emotions2labels)
    test_data = exp_1_load_dataloader(test_data_list, emotions2labels)

    device = "cuda:0"
    device = torch.device(device)

    text_encoder = exp_12_load_model(
        labels=len(all_emotions),
    )

    text_encoder = text_encoder.to(device)
    model_name = "sentence-transformers/all-mpnet-base-v2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    print(text_encoder)

    training_args = TrainingArguments(
        output_dir="semeval/experiments/kosenko/text_encoders/train_results/",
        evaluation_strategy="epoch",
        num_train_epochs=10,
        save_strategy="epoch",
        save_total_limit=1,
        # report_to="none",
        report_to="wandb",
        logging_steps=5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        gradient_accumulation_steps=8,
        bf16=True,
        remove_unused_columns=False,
        metric_for_best_model="eval_f1_score",
    )

    trainer = CustomTrainer(
        model=text_encoder,
        args=training_args,
        train_dataset=training_data,
        eval_dataset=test_data,
        compute_metrics=compute_metrics,
    )
    trainer.train()
