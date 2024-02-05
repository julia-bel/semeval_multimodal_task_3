import os

# os.environ["WANDB_PROJECT"] = "semeval_cause_classification"
# os.environ["CUDA_VISIBLE_DEVICES"] = "3"

from semeval.experiments.kosenko.language_bind.languagebind_classification_video_text import (
    exp_1_get_modality_config,
)


# os.environ["WANDB_PROJECT"] = "semeval_emotion_classification"
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


from semeval.experiments.kosenko.language_bind.custom_bert import (
    BertForCauseAnswering,
)
from transformers import BertTokenizer, AutoConfig


def cause_collate_fn(item):
    new_item = {}
    for line in item:
        for key in line.keys():
            if key in new_item:
                new_item[key].append(line[key])
            else:
                new_item[key] = [line[key]]

    return new_item


class CustomTrainer(Trainer):
    def get_train_dataloader(self):
        train_dataset = self.train_dataset
        return DataLoader(
            train_dataset,
            batch_size=self.args.per_device_train_batch_size,
            shuffle=True,
            collate_fn=cause_collate_fn,
        )

    def get_eval_dataloader(self, eval_dataset):
        return DataLoader(
            self.eval_dataset,
            batch_size=self.args.per_device_eval_batch_size,
            shuffle=False,
            collate_fn=cause_collate_fn,
        )


class CustomTrainerCausal(CustomTrainer):
    def compute_loss(
        self,
        model,
        inputs,
        return_outputs=False,
    ):
        model.train()
        # forward pass
        cause = model(inputs)
        cause_label = []
        for label in inputs["label"]:
            if label == -1:
                # point to CLS token
                cause_label.append(0)
            else:
                # point to real token
                true_pos = (label + 1) * 2
                cause_label.append(true_pos)

        cause_label = torch.tensor(
            cause_label,
            device=device,
        )

        loss_func = torch.nn.CrossEntropyLoss()
        cause_loss = loss_func(cause, cause_label)
        loss = cause_loss
        return (
            (
                loss,
                {
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
            cause_label = []
            for label in inputs["label"]:
                if label == -1:
                    # point to CLS token
                    cause_label.append(0)
                else:
                    # point to real token
                    true_pos = (label + 1) * 2
                    cause_label.append(true_pos)
            cause_label = torch.tensor(
                cause_label,
                device=device,
            )

        if prediction_loss_only:
            return (loss, None, None)

        return (loss, outputs, cause_label)


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
    pred_causal = predictions["cause"].argmax(-1)
    labels = eval_preds.label_ids

    f1_score_causal = f1_score(
        labels,
        pred_causal,
        average="macro",
    )
    return {
        "f1_score_causal": f1_score_causal,
    }


class BERTCauseConversationsDataset(Dataset):
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

        turn["video_base_path"] = self.base_video_path

        return turn


class CausePredictor(torch.nn.Module):
    def __init__(
        self,
        clip_type=None,
        labels=None,
    ):
        super().__init__()
        config = AutoConfig.from_pretrained("bert-base-uncased")
        config.num_hidden_layers = 2

        self.bert_model = BertForCauseAnswering._from_config(config)
        self.model = LanguageBind(
            clip_type=clip_type,
            cache_dir="/code/cache_dir",
        )
        self.model.eval()

        self.PAD = 0
        self.CLS = 1
        self.SEP = 2
        self.BOS = 3

        self.special_embeddings = torch.nn.Embedding(
            4,
            768,
            padding_idx=0,
        )

        self.device = "cuda"

    def get_embeddings(self, batch):
        embeddings = []

        for i in range(len(batch["conversation"])):
            conversation = batch["conversation"][i]

            video = []
            language = []
            base_video_path = batch["video_base_path"][0]
            for utterance in conversation:
                video_path = f"{base_video_path}/{utterance['video_name']}"
                language_utt = utterance["text"]
                video.append(video_path)
                language.append(language_utt)

            inputs = {
                "video": to_device(
                    modality_transform["video"](video),
                    device,
                ),
                "language": to_device(
                    tokenizer(
                        language,
                        max_length=77,
                        padding="max_length",
                        truncation=True,
                        return_tensors="pt",
                    ),
                    device,
                ),
            }
            with torch.no_grad():
                with torch.autocast(device_type="cuda"):
                    emb = self.model(inputs)
            embeddings.append(emb)
        return embeddings

    def preprocess_embeddings(self, batch):
        device = self.device
        embeddings = self.get_embeddings(batch=batch)
        new_embeddings = []
        max_len = 0
        for emb_pos, emb in enumerate(embeddings):
            new_vector = []
            new_vector.append(
                self.special_embeddings(
                    torch.tensor(
                        [self.CLS],
                        device=device,
                    )
                )
            )

            for i in range(emb["video"].shape[0]):
                video_vec = emb["video"][i]
                lang_vec = emb["language"][i]
                new_vector.append(video_vec)
                new_vector.append(lang_vec)

            new_vector.append(
                self.special_embeddings(
                    torch.tensor(
                        [self.SEP],
                        device=device,
                    )
                )
            )

            # эмоция которую у которой мы хотим найти причину
            new_vector.append(
                emb["video"][batch["emotion_pos"][emb_pos]],
            )
            new_vector.append(
                emb["language"][batch["emotion_pos"][emb_pos]],
            )

            new_vector.append(
                self.special_embeddings(
                    torch.tensor(
                        [self.SEP],
                        device=device,
                    )
                )
            )

            # вероятная причина
            new_vector.append(
                emb["video"][batch["cause_pos"][emb_pos]],
            )
            new_vector.append(
                emb["language"][batch["cause_pos"][emb_pos]],
            )

            new_vector.append(
                self.special_embeddings(
                    torch.tensor(
                        [self.BOS],
                        device=device,
                    )
                )
            )
            max_len = max(max_len, len(new_vector))
            new_vector = torch.vstack(new_vector)
            new_embeddings.append(new_vector)

        return new_embeddings, max_len

    def pad_vectors(self, embeddings, max_len):
        attention_mask = []
        for i in range(len(embeddings)):
            emb_len = embeddings[i].shape[0]
            emb_diff_amount = max_len - emb_len
            if emb_diff_amount != 0:
                emb_pad = self.special_embeddings(
                    torch.tensor(
                        [self.PAD] * emb_diff_amount,
                        device=device,
                    )
                )
                embeddings[i] = torch.cat(
                    [
                        embeddings[i],
                        emb_pad,
                    ],
                    dim=0,
                )
                attention_mask.append([1] * emb_len + [0] * emb_diff_amount)
            else:
                attention_mask.append([1] * emb_len)

            embeddings[i] = embeddings[i].unsqueeze(0)

        attention_mask = torch.tensor(
            attention_mask,
            device=self.device,
        )

        return embeddings, attention_mask

    def forward(self, batch):
        embeddings, max_len = self.preprocess_embeddings(
            batch=batch,
        )
        embeddings, attention_mask = self.pad_vectors(
            embeddings=embeddings,
            max_len=max_len,
        )
        embeddings = torch.vstack(embeddings)
        cause_logits = self.bert_model(
            inputs_embeds=embeddings,
            attention_mask=attention_mask,
        )
        return cause_logits.start_logits


def exp_11_load_model(labels, clip_type):
    text_video_classif = CausePredictor(
        labels=labels,
        clip_type=clip_type,
    )

    return text_video_classif


def exp_11_load_dataset():
    dataset = load_dataset("dim/semeval_subtask2_conversations")
    dataset_train = get_bert_cause_dataset(dataset=dataset["train"])
    dataset_test = get_bert_cause_dataset(dataset=dataset["test"])

    return {
        "train": dataset_train,
        "test": dataset_test,
    }


def get_bert_cause_dataset(dataset):
    new_dataset = []

    for item in dataset:
        conversation = item["conversation"]
        # print(item)
        positive_pairs = []

        for cause in item["emotion-cause_pairs"]:
            emotion_pos = int(cause[0].split("_")[0]) - 1
            emotion = cause[0].split("_")[1]
            cause_pos = int(cause[1]) - 1
            # print(emotion_pos, emotion, cause_pos, cause)
            new_dataset.append(
                {
                    "conversation": conversation,
                    "emotion_pos": emotion_pos,
                    "cause_pos": cause_pos,
                    "emotion_label": emotion,
                    "label": cause_pos,
                }
            )
            positive_pairs.append((emotion_pos, cause_pos))

        positive_pairs = set(positive_pairs)
        negative_pairs = []
        for pos_i in range(len(conversation)):
            for pos_j in range(len(conversation)):
                pair = (pos_i, pos_j)
                if not pair in positive_pairs:
                    negative_pairs.append(pair)

        if len(negative_pairs) > len(positive_pairs):
            negative_pairs = random.sample(negative_pairs, len(positive_pairs))

        for pair in negative_pairs:
            emotion = conversation[pair[0]]["emotion"]
            new_dataset.append(
                {
                    "conversation": conversation,
                    "emotion_pos": pair[0],
                    "cause_pos": pair[1],
                    "emotion_label": emotion,
                    "label": -1,
                }
            )

    return new_dataset


def exp_11_load_dataloader(dataset):
    return BERTCauseConversationsDataset(conversations=dataset)


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

    dataset = exp_11_load_dataset()
    training_data_list, test_data_list = dataset["train"], dataset["test"]
    # training_data_list = training_data_list[:2]
    # test_data_list = test_data_list[:2]
    training_data = exp_11_load_dataloader(training_data_list)
    test_data = exp_11_load_dataloader(test_data_list)

    device = "cuda:0"
    device = torch.device(device)
    clip_type = {
        "video": "LanguageBind_Video_FT",
    }

    text_video_classif = exp_11_load_model(
        labels=len(all_emotions),
        clip_type=clip_type,
    )

    text_video_classif = text_video_classif.to(device)
    pretrained_ckpt = f"LanguageBind/LanguageBind_Image"
    tokenizer = LanguageBindImageTokenizer.from_pretrained(
        pretrained_ckpt,
        cache_dir="/code/cache_dir/tokenizer_cache_dir",
    )
    modality_config = exp_1_get_modality_config(text_video_classif)
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
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=4,
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
