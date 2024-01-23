import os

from semeval.experiments.kosenko.language_bind.languagebind_classification_video_text import (
    exp_4_get_modality_config,
)


os.environ["WANDB_PROJECT"] = "semeval_emotion_classification"
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
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
from peft import (
    get_peft_config,
    PeftModel,
    PeftConfig,
    get_peft_model,
    LoraConfig,
    TaskType,
)


class CustomTrainer(Trainer):
    def compute_loss(
        self,
        model,
        inputs,
        return_outputs=False,
    ):
        custom_inputs = self.get_inputs(inputs=inputs)
        # forward pass
        outputs = model(custom_inputs)
        label = inputs["label"].to(device)
        loss_func = torch.nn.CrossEntropyLoss()
        loss = loss_func(outputs, label)
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

    def get_inputs(self, inputs):
        video_paths = [
            f"{base_path}/{video_path}"
            for base_path, video_path in zip(
                inputs["video_base_path"], inputs["video_name"]
            )
        ]
        audio_paths = [
            f"{base_path}/{audio_name}"
            for base_path, audio_name in zip(
                inputs["video_base_path"], inputs["audio_name"]
            )
        ]
        custom_inputs = {
            "video": to_device(
                modality_transform["video"](video_paths),
                device,
            ),
            "audio": to_device(
                modality_transform["audio"](audio_paths),
                device,
            ),
            "language": to_device(
                tokenizer(
                    inputs["text"],
                    max_length=77,
                    padding="max_length",
                    truncation=True,
                    return_tensors="pt",
                ),
                device,
            ),
        }
        return custom_inputs

    def prediction_step(
        self,
        model,
        inputs,
        prediction_loss_only,
        ignore_keys=None,
    ):
        model.eval()
        with torch.no_grad():
            # forward pass
            custom_inputs = self.get_inputs(inputs=inputs)
            outputs = model(custom_inputs)
            label = inputs["label"].to(device)
            loss_func = torch.nn.CrossEntropyLoss()
            loss = loss_func(outputs, label)
        if prediction_loss_only:
            return (loss, None, None)

        return (loss, outputs, label)


def compute_metrics(eval_preds):
    # metric = evaluate.load("glue", "mrpc")
    # print(eval_preds)
    # logits, labels = eval_preds
    predictions = eval_preds.predictions.argmax(-1)
    labels = eval_preds.label_ids
    # return metric.compute(predictions=predictions, references=labels)
    f1_score_result = f1_score(
        labels,
        predictions,
        average="macro",
    )
    return {
        "f1_score": f1_score_result,
    }


class VideoAudioTextClassif(torch.nn.Module):
    def __init__(self, labels=2, clip_type=None):
        super().__init__()
        self.model = LanguageBind(
            clip_type=clip_type,
            cache_dir="/code/cache_dir",
        )
        self.linear = torch.nn.Linear(
            768 * 3,
            labels,
            bias=False,
        )

    def forward(self, x):
        result = self.model(x)
        # print(result)
        features = torch.cat(
            [
                result["video"],
                result["audio"],
                result["language"],
            ],
            dim=-1,
        )
        result = self.linear(features)
        return result


def exp_2_load_model(labels, clip_type):
    text_video_classif = VideoAudioTextClassif(
        labels=labels,
        clip_type=clip_type,
    )
    return text_video_classif


def exp_6_load_model(labels, clip_type):
    text_video_classif = VideoAudioTextClassif(
        labels=labels,
        clip_type=clip_type,
    )
    peft_config = LoraConfig(
        inference_mode=False,
        r=8,
        lora_alpha=8,
        lora_dropout=0.0,
        bias="all",
        target_modules=[
            "k_proj",
            "v_proj",
            "q_proj",
            "out_proj",
            "fc1",
            "fc2",
        ],
    )
    # text_video_classif = PeftModel.from_pretrained(
    #     text_video_classif,
    #     "semeval/experiments/kosenko/language_bind/train_results/checkpoint-195",
    #     is_trainable=True,
    # )
    text_video_classif = get_peft_model(text_video_classif, peft_config)
    text_video_classif.print_trainable_parameters()
    text_video_classif.config = None
    # text_video_classif.load_adapter(
    #     model_id="semeval/experiments/kosenko/language_bind/train_results/checkpoint-979"
    # )
    return text_video_classif


class ConversationsDataset(Dataset):
    def __init__(
        self,
        conversations,
        base_video_path="/code/SemEval-2024_Task3/training_data/train",
    ):
        self.conversations = conversations

        self.base_video_path = base_video_path

    def __len__(self):
        return len(self.conversations)

    def __getitem__(self, idx):
        turn = self.conversations[idx]

        turn["audio_name"] = turn["video_name"].replace(".mp4", ".wav")
        turn["video_base_path"] = self.base_video_path
        # print(video_path)
        turn["label"] = emotions2labels[turn["emotion"]]

        return turn


def exp_2_get_modality_config(model):
    return model.model.modality_config


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

    dataset = load_dataset("dim/SemEval_training_data_emotions")
    training_data_list, test_data_list = dataset["train"], dataset["test"]
    # training_data_list = training_data_list[:1000]
    # test_data_list = test_data_list[:200]
    training_data = ConversationsDataset(conversations=training_data_list)
    test_data = ConversationsDataset(conversations=test_data_list)

    device = "cuda:0"
    device = torch.device(device)
    clip_type = {
        "video": "LanguageBind_Video_FT",
        "audio": "LanguageBind_Audio_FT",
    }

    # text_video_classif = exp_2_load_model(
    #     labels=len(all_emotions),
    #     clip_type=clip_type,
    # )
    text_video_classif = exp_6_load_model(
        labels=len(all_emotions),
        clip_type=clip_type,
    )

    text_video_classif = text_video_classif.to(device)
    pretrained_ckpt = f"LanguageBind/LanguageBind_Image"
    tokenizer = LanguageBindImageTokenizer.from_pretrained(
        pretrained_ckpt,
        cache_dir="/code/cache_dir/tokenizer_cache_dir",
    )
    # modality_config = exp_2_get_modality_config(text_video_classif)
    modality_config = exp_4_get_modality_config(text_video_classif)
    modality_transform = {
        c: transform_dict[c](modality_config[c]) for c in clip_type.keys()
    }

    training_args = TrainingArguments(
        output_dir="semeval/experiments/kosenko/language_bind/train_results/",
        evaluation_strategy="epoch",
        num_train_epochs=10,
        save_strategy="epoch",
        save_total_limit=1,
        report_to="wandb",
        # report_to="none",
        logging_steps=5,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=16,
        bf16=True,
        remove_unused_columns=False,
    )

    trainer = CustomTrainer(
        model=text_video_classif,
        args=training_args,
        train_dataset=training_data,
        eval_dataset=test_data,
        compute_metrics=compute_metrics,
    )
    trainer.train(
        # resume_from_checkpoint="semeval/experiments/kosenko/language_bind/train_results/checkpoint-979"
    )
