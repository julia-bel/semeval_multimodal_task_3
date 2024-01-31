import os


# os.environ["WANDB_PROJECT"] = "semeval_emotion_classification"
os.environ["WANDB_PROJECT"] = "semeval_cause_classification"
# os.environ["CUDA_VISIBLE_DEVICES"] = "3"
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


def cause_collate_fn(item):
    return item


class CustomTrainer(Trainer):
    def compute_loss(
        self,
        model,
        inputs,
        return_outputs=False,
    ):
        video_paths = [
            f"{base_path}/{video_path}"
            for base_path, video_path in zip(
                inputs["video_base_path"], inputs["video_name"]
            )
        ]
        custom_inputs = {
            "video": to_device(
                modality_transform["video"](video_paths),
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

    def prediction_step(
        self,
        model,
        inputs,
        prediction_loss_only,
        ignore_keys=None,
    ):
        model.eval()
        with torch.no_grad():
            video_paths = [
                f"{base_path}/{video_path}"
                for base_path, video_path in zip(
                    inputs["video_base_path"], inputs["video_name"]
                )
            ]
            custom_inputs = {
                "video": to_device(
                    modality_transform["video"](video_paths),
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

            # forward pass
            outputs = model(custom_inputs)
            label = inputs["label"].to(device)
            loss_func = torch.nn.CrossEntropyLoss()
            loss = loss_func(outputs, label)
        if prediction_loss_only:
            return (loss, None, None)

        return (loss, outputs, label)


class CustomTrainerCausal(CustomTrainer):
    def compute_loss(
        self,
        model,
        inputs,
        return_outputs=False,
    ):
        model.train()
        initial_video_paths = [
            f"{base_path}/{video_path}"
            for base_path, video_path in zip(
                inputs["video_base_path"], inputs["initial_video_name"]
            )
        ]
        causal_video_paths = [
            f"{base_path}/{video_path}"
            for base_path, video_path in zip(
                inputs["video_base_path"], inputs["cause_video_name"]
            )
        ]
        custom_inputs = {
            "initial_video": to_device(
                modality_transform["video"](initial_video_paths),
                device,
            ),
            "cause_video": to_device(
                modality_transform["video"](causal_video_paths),
                device,
            ),
            "initial_language": to_device(
                tokenizer(
                    inputs["initial_text"],
                    max_length=77,
                    padding="max_length",
                    truncation=True,
                    return_tensors="pt",
                ),
                device,
            ),
            "cause_language": to_device(
                tokenizer(
                    inputs["cause_text"],
                    max_length=77,
                    padding="max_length",
                    truncation=True,
                    return_tensors="pt",
                ),
                device,
            ),
        }

        # forward pass
        emotion, cause = model(custom_inputs)
        emotion_label = inputs["emotion"].to(device)
        cause_label = inputs["cause"].to(device)

        loss_func = torch.nn.CrossEntropyLoss()
        emotion_loss = loss_func(emotion, emotion_label)
        cause_loss = loss_func(cause, cause_label)
        loss = emotion_loss + cause_loss
        return (
            (
                loss,
                {
                    "emotion": emotion,
                    "cause": cause,
                },
            )
            if return_outputs
            else loss
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
            loss, outputs = self.compute_loss(model, inputs, return_outputs=True)
            label = {
                "emotion": inputs["emotion"].to(device),
                "cause": inputs["cause"].to(device),
            }

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


def compute_metrics_cause(eval_preds):
    predictions = eval_preds.predictions
    pred_emotion = predictions["emotion"].argmax(-1)
    pred_causal = predictions["cause"].argmax(-1)
    labels = eval_preds.label_ids
    f1_score_emotion = f1_score(
        labels["emotion"],
        pred_emotion,
        average="macro",
    )
    f1_score_causal = f1_score(
        labels["cause"],
        pred_causal,
        average="macro",
    )
    return {
        "f1_score_emotion": f1_score_emotion,
        "f1_score_causal": f1_score_causal,
    }


class VideoTextClassif(torch.nn.Module):
    def __init__(self, labels=2, clip_type=None):
        super().__init__()
        self.model = LanguageBind(
            clip_type=clip_type,
            cache_dir="/code/cache_dir",
        )
        self.linear = torch.nn.Linear(
            768 * 2,
            labels,
            bias=False,
        )

    def forward(self, x):
        result = self.model(x)
        # print(result)
        features = torch.cat(
            [
                result["video"],
                result["language"],
            ],
            dim=-1,
        )
        result = self.linear(features)
        return result


class VideoTextClassif2(torch.nn.Module):
    def __init__(self, labels=2, clip_type=None):
        super().__init__()
        self.model = LanguageBind(
            clip_type=clip_type,
            cache_dir="/code/cache_dir",
        )
        # чтобы векторы с видео модели, совпали с векторами из языковой
        self.video_projection = torch.nn.Linear(
            1024,
            768,
            bias=False,
        )
        self.multihead_attn = nn.MultiheadAttention(768, 4)

        self.linear = torch.nn.Linear(
            768,
            labels,
        )

    def forward(self, x):
        result = self.model(x)
        language_hidden_state = result["language_encoder"]
        batch_size = language_hidden_state.shape[0]
        frames = 8
        video_hidden_state = result["video_encoder"][:, 0, :]
        video_hidden_state = video_hidden_state.reshape(batch_size, frames, -1)
        video_hidden_state = self.video_projection(video_hidden_state)
        total_hidden_state = torch.cat(
            [video_hidden_state, language_hidden_state],
            dim=1,
        )
        total_hidden_state = language_hidden_state
        attn_output, attn_output_weights = self.multihead_attn(
            total_hidden_state,
            total_hidden_state,
            total_hidden_state,
        )
        feature_vector = attn_output.mean(1)
        result = self.linear(feature_vector)
        return result


class CauseVideoTextClassif(torch.nn.Module):
    def __init__(self, labels=2, clip_type=None):
        super().__init__()
        self.model = LanguageBind(
            clip_type=clip_type,
            cache_dir="/code/cache_dir",
        )
        self.emotion_classif = torch.nn.Linear(
            768 * 4,
            labels,
        )
        self.cause_classif = torch.nn.Linear(
            768 * 4,
            2,
        )

    def forward(self, x):
        initial_result = self.model(
            {
                "video": x["initial_video"],
                "language": x["initial_language"],
            }
        )
        cause_result = self.model(
            {
                "video": x["cause_video"],
                "language": x["cause_language"],
            }
        )

        features = torch.cat(
            [
                initial_result["video"],
                initial_result["language"],
                cause_result["video"],
                cause_result["language"],
            ],
            dim=-1,
        )
        emotion = self.emotion_classif(features)
        cause = self.cause_classif(features)
        return emotion, cause


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

        turn["video_name"] = turn["video_name"]
        turn["video_base_path"] = self.base_video_path
        # print(video_path)
        turn["label"] = emotions2labels[turn["emotion"]]


class CauseConversationsDataset(Dataset):
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

        turn["initial_video_name"] = turn["initial"]["video_name"]
        turn["initial_text"] = turn["initial"]["text"]

        turn["cause_video_name"] = turn["causal"]["video_name"]
        turn["cause_text"] = turn["causal"]["text"]

        turn["video_base_path"] = self.base_video_path
        # print(video_path)
        turn["emotion"] = emotions2labels[turn["initial"]["emotion"]]
        turn["cause"] = turn["label"]

        return turn


def exp_1_load_model(labels, clip_type):
    text_video_classif = VideoTextClassif(
        labels=labels,
        clip_type=clip_type,
    )
    return text_video_classif


def exp_7_load_model(labels, clip_type):
    text_video_classif = VideoTextClassif2(
        labels=labels,
        clip_type=clip_type,
    )
    for param in text_video_classif.named_parameters():
        if "model" in param[0]:
            param[1].requires_grad_(False)
    return text_video_classif


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def exp_3_load_model(labels, clip_type):
    text_video_classif = VideoTextClassif(
        labels=labels,
        clip_type=clip_type,
    )
    for param in text_video_classif.named_parameters():
        if "model" in param[0]:
            param[1].requires_grad_(False)

    print(f"Trainable params: {count_parameters(text_video_classif)}")
    return text_video_classif


def exp_4_load_model(labels, clip_type):
    text_video_classif = VideoTextClassif(
        labels=labels,
        clip_type=clip_type,
    )
    peft_config = LoraConfig(
        inference_mode=False,
        r=16,
        lora_alpha=16,
        lora_dropout=0.1,
        bias="all",
        target_modules=[
            "k_proj",
            "v_proj",
            "q_proj",
            "out_proj",
        ],
    )
    text_video_classif = get_peft_model(text_video_classif, peft_config)
    text_video_classif.print_trainable_parameters()
    text_video_classif.config = None
    return text_video_classif


def exp_5_load_model(labels, clip_type):
    text_video_classif = VideoTextClassif(
        labels=labels,
        clip_type=clip_type,
    )
    peft_config = LoraConfig(
        inference_mode=False,
        r=16,
        lora_alpha=16,
        lora_dropout=0.1,
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
    text_video_classif = get_peft_model(text_video_classif, peft_config)
    text_video_classif.print_trainable_parameters()
    text_video_classif.config = None
    return text_video_classif


def exp_9_load_model(labels, clip_type):
    text_video_classif = CauseVideoTextClassif(
        labels=labels,
        clip_type=clip_type,
    )

    return text_video_classif


def exp_10_load_model(labels, clip_type):
    text_video_classif = CauseVideoTextClassif(
        labels=labels,
        clip_type=clip_type,
    )

    peft_config = LoraConfig(
        inference_mode=False,
        r=16,
        lora_alpha=16,
        lora_dropout=0.1,
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
    text_video_classif = get_peft_model(text_video_classif, peft_config)
    text_video_classif.print_trainable_parameters()
    text_video_classif.config = None
    return text_video_classif


def exp_1_get_modality_config(model):
    return model.model.modality_config


def exp_4_get_modality_config(model):
    return model.model.model.modality_config


def exp_1_load_dataloader(dataset):
    return ConversationsDataset(conversations=dataset)


def exp_9_load_dataloader(dataset):
    return CauseConversationsDataset(conversations=dataset)


def exp_1_load_dataset():
    return load_dataset("dim/SemEval_training_data_emotions")


def exp_9_generate_dataset(dataset):
    new_dataset = []
    for i, conversation in enumerate(dataset):
        cause_pairs = conversation["emotion-cause_pairs"]
        # print(i, cause_pairs)
        positive_pairs = []
        for cause_pair in cause_pairs:
            positive_example = {"initial": {}, "causal": {}, "label": 1}
            initial_emotion_pos = int(cause_pair[0].split("_")[0]) - 1
            initial_emotion = conversation["conversation"][initial_emotion_pos]
            cause_emotion_pos = int(cause_pair[1]) - 1
            cause_emotion = conversation["conversation"][cause_emotion_pos]
            positive_example["initial"] = initial_emotion
            positive_example["causal"] = cause_emotion

            positive_pairs.append((initial_emotion_pos, cause_emotion_pos))
            new_dataset.append(positive_example)

        # new_dataset.extend(positive_pairs)
        positive_pairs = set(positive_pairs)
        negative_pairs = []
        for pos_i in range(len(conversation["conversation"])):
            for pos_j in range(len(conversation["conversation"])):
                pair = (pos_i, pos_j)
                if not pair in positive_pairs:
                    negative_pairs.append(pair)
        # print(len(negative_pairs), positive_pairs)

        if len(negative_pairs) > len(positive_pairs):
            negative_pairs = random.sample(negative_pairs, len(positive_pairs))

        for pair in negative_pairs:
            negative_example = {"initial": {}, "causal": {}, "label": 0}
            negative_example["initial"] = conversation["conversation"][pair[0]]
            negative_example["causal"] = conversation["conversation"][pair[1]]
            new_dataset.append(negative_example)

        # print("==" * 10)
        # print("==" * 10)
        # new_dataset.extend(negative_pairs)

    return new_dataset


def exp_9_load_dataset():
    dataset = load_dataset("dim/semeval_subtask2_conversations")
    train = exp_9_generate_dataset(dataset=dataset["train"])
    test = exp_9_generate_dataset(dataset=dataset["test"])
    return {
        "train": train,
        "test": test,
    }


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

    dataset = exp_9_load_dataset()
    training_data_list, test_data_list = dataset["train"], dataset["test"]
    # training_data_list = training_data_list[:2]
    # test_data_list = test_data_list[:2]
    training_data = exp_9_load_dataloader(training_data_list)
    test_data = exp_9_load_dataloader(test_data_list)

    device = "cuda:0"
    device = torch.device(device)
    clip_type = {
        "video": "LanguageBind_Video_FT",
    }

    # text_video_classif = exp_1_load_model(
    #     labels=len(all_emotions),
    #     clip_type=clip_type,
    # )
    text_video_classif = exp_10_load_model(
        labels=len(all_emotions),
        clip_type=clip_type,
    )

    text_video_classif = text_video_classif.to(device)
    pretrained_ckpt = f"LanguageBind/LanguageBind_Image"
    tokenizer = LanguageBindImageTokenizer.from_pretrained(
        pretrained_ckpt,
        cache_dir="/code/cache_dir/tokenizer_cache_dir",
    )
    modality_config = exp_4_get_modality_config(text_video_classif)
    # modality_config = exp_1_get_modality_config(text_video_classif)
    modality_transform = {
        c: transform_dict[c](modality_config[c]) for c in clip_type.keys()
    }
    print(text_video_classif)

    training_args = TrainingArguments(
        output_dir="semeval/experiments/kosenko/language_bind/train_results/",
        evaluation_strategy="epoch",
        num_train_epochs=10,
        save_strategy="epoch",
        save_total_limit=1,
        # report_to="none",
        report_to="wandb",
        logging_steps=5,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=8,
        bf16=True,
        remove_unused_columns=False,
        metric_for_best_model="eval_f1_score_causal",
    )

    # trainer = CustomTrainer(
    trainer = CustomTrainerCausal(
        model=text_video_classif,
        args=training_args,
        train_dataset=training_data,
        eval_dataset=test_data,
        compute_metrics=compute_metrics_cause,
    )
    trainer.train()
