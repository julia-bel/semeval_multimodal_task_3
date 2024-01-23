import random

import torch
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset
from transformers import LlamaTokenizer
from video_llama.models.ImageBind.data import load_and_transform_audio_data
from video_llama.processors import AlproVideoEvalProcessor, AlproVideoTrainProcessor
from video_llama.processors.video_processor import load_video

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


class EmotionDataset(Dataset):
    def __init__(
        self,
        data_name="dim/SemEval_training_data_emotions",
        root="/code/data/video_with_audio",
        split="train",
        num_frames=8,
        resize_size=224,
        tokenizer_name="/code/SemEvalParticipants/semeval/experiments/belikova/videollama/ckpt/llama-2-13b-chat-hf",
    ):
        self.root = root
        self.annotation = load_dataset(data_name, split=split)
        self.num_frames = num_frames
        self.resize_size = resize_size
        if split == "train":
            self.transform = AlproVideoTrainProcessor(
                image_size=resize_size,
                n_frms=num_frames,
            ).transform
        else:
            self.transform = AlproVideoEvalProcessor(
                image_size=resize_size,
                n_frms=num_frames,
            ).transform
        self.tokenizer = LlamaTokenizer.from_pretrained(tokenizer_name, use_fast=False)
        self.tokenizer.pad_token = self.tokenizer.unk_token

    def __len__(self):
        return len(self.annotation)

    def __getitem__(self, index, num_retries=10, device="cpu"):
        result = {}
        for _ in range(num_retries):
            sample = self.annotation[index]
            video_path = "/".join([self.root, sample["video_name"]])
            try:
                result["video"] = self.transform(
                    load_video(
                        video_path=video_path,
                        n_frms=self.num_frames,
                        height=self.resize_size,
                        width=self.resize_size,
                        sampling="uniform",
                        return_msg=False,
                    )
                )
                result["text"] = self.tokenizer(
                    sample["text"],
                    return_tensors="pt",
                    padding="longest",
                    max_length=512,
                    truncation=True,
                ).input_ids[0]
                result["audio"] = load_and_transform_audio_data(
                    [video_path],
                    device=device,
                    clips_per_video=self.num_frames,
                )[0]
                result["label"] = emotions2labels[sample["emotion"]]
                assert (
                    result["video"].shape[1]
                    == self.num_frames
                    == result["audio"].shape[0]
                )
            except Exception:
                index = random.randint(0, len(self) - 1)
                continue
            break
        else:
            raise RuntimeError(f"Failed to fetch sample after {num_retries} retries.")
        return result

    def collater(self, instances):
        text_ids = [instance["text"] for instance in instances]
        text_ids = torch.nn.utils.rnn.pad_sequence(
            text_ids,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id,
        )

        batch = {
            "video": torch.stack([instance["video"] for instance in instances]),
            "text": text_ids,
            "audio": torch.stack([instance["audio"] for instance in instances]),
            "label": torch.tensor([instance["label"] for instance in instances]),
        }

        return batch


if __name__ == "__main__":
    train_dataset = EmotionDataset(split="train")
    val_dataset = EmotionDataset(split="test")
    train_loader = DataLoader(
        train_dataset,
        batch_size=32,
        shuffle=True,
        num_workers=4,
        collate_fn=train_dataset.collater,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=32, num_workers=4, collate_fn=val_dataset.collater
    )
