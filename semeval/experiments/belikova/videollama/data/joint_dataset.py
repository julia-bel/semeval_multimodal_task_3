import re

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


class EmotionCausalDataset(Dataset):
    def __init__(
        self,
        data_name="dim/semeval_subtask2_conversations",
        root="semeval/data/video_with_audio",
        split="train",
        num_frames=8,
        resize_size=224,
        max_text_len=340,
        modalities=("video", "text"),
        tokenizer_name="semeval/experiments/belikova/videollama/ckpt/llama-2-13b-chat-hf",
    ):
        self.root = root
        self.annotation = load_dataset(data_name, split=split)
        self.modalities = modalities
        self.num_frames = num_frames
        self.resize_size = resize_size
        self.max_text_len = max_text_len
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

    def __pad_video(self, video, device="cpu"):
        if video.shape[1] != self.num_frames:
            c, f, h, w = video.shape
            zero_pad = torch.zeros(c, self.num_frames - f, h, w, device=device)
            video = torch.cat((video, zero_pad), dim=1)
        return video

    def __pad_audio(self, audio, device="cpu"):
        if audio.shape[0] != self.num_frames:
            f, c, h, w = audio.shape
            zero_pad = torch.zeros(self.num_frames - f, c, h, w, device=device)
            audio = torch.cat((audio, zero_pad), dim=1)
        return audio

    def __get_utterances(
        self,
        conversation,
        device="cpu",
    ):
        result = {m: [] for m in self.modalities}
        result["emotion"] = []
        for sample in conversation:
            if "video" in self.modalities:
                video_path = "/".join([self.root, sample["video_name"]])
                result["video"].append(
                    self.__pad_video(
                        self.transform(
                            load_video(
                                video_path=video_path,
                                n_frms=self.num_frames,
                                height=self.resize_size,
                                width=self.resize_size,
                                sampling="uniform",
                                return_msg=False,
                            )
                        ),
                        device,
                    )
                )
            if "audio" in self.modalities:
                result["audio"].append(
                    self.__pad_audio(
                        load_and_transform_audio_data(
                            [video_path],
                            device=device,
                            clips_per_video=self.num_frames,
                        )[0],
                        device,
                    )
                )
            if "text" in self.modalities:
                result["text"].append(
                    self.tokenizer(
                        sample["text"],
                        return_tensors="pt",
                        padding="longest",
                        max_length=512,
                        truncation=True,
                    ).input_ids[0]
                )
            result["emotion"].append(emotions2labels[sample["emotion"]])

        return result

    def __get_id(self, name):
        return int(re.sub("[^1-9]*", "", name)) - 1

    def __getitem__(self, index, device="cpu"):
        # result = {
        #     "video": ...,
        #     "text": ...,
        #     "audio": ...,
        #     "causal_pair": ...,
        #     "emotion": ...,
        # }

        result = {}
        sample = self.annotation[index]
        cause_pair = sample["emotion-cause_pairs"]
        conversation = sample["conversation"]
        utterances = self.__get_utterances(conversation, device)
        if "text" in self.modalities:
            padded_text = torch.nn.utils.rnn.pad_sequence(
                utterances["text"],
                batch_first=True,
                padding_value=self.tokenizer.pad_token_id,
            )
            result["text"] = torch.nn.functional.pad(
                padded_text,
                (0, self.max_text_len - padded_text.shape[1]),
                value=self.tokenizer.pad_token_id,
            )
        if "video" in self.modalities:
            result["video"] = torch.stack(utterances["video"])
        if "audio" in self.modalities:
            result["audio"] = torch.stack(utterances["audio"])
        result["emotion"] = torch.tensor(utterances["emotion"])
        result["cause_pair"] = torch.LongTensor(
            [[self.__get_id(p[0]), self.__get_id(p[1])] for p in cause_pair]
        ).t()

        return result

    def collater(self, instances):
        # batch = {
        #     "video": ...,
        #     "text": ...,
        #     "audio": ...,
        #     "utterance_length": ...,
        #     "causal_relationship": ...,
        #     "emotion": ...,
        # }
        batch = {}

        utterance_lengths = torch.tensor([len(i["emotion"]) for i in instances])
        max_utterance_num = max(utterance_lengths)
        range_tensor = torch.arange(max_utterance_num).unsqueeze(0)
        range_tensor = range_tensor.expand(len(utterance_lengths), max_utterance_num)
        expanded_lengths = utterance_lengths.unsqueeze(1).expand_as(range_tensor)
        padding_mask = (range_tensor < expanded_lengths).float()
        batch["utterance_length"] = padding_mask

        cause_pairs = []
        for instance in instances:
            cause_pair = instance["cause_pair"]
            if cause_pair.dim() > 1:
                values = torch.ones(cause_pair.size(1))
                sparse_tensor = torch.sparse_coo_tensor(
                    cause_pair, values, (max_utterance_num, max_utterance_num)
                )
                cause_pairs.append(torch.clamp(sparse_tensor.to_dense(), min=0, max=1))
            else:
                cause_pairs.append(torch.zeros(max_utterance_num, max_utterance_num))
        mask = padding_mask[:, :, None] @ padding_mask[:, None, :] - 1
        batch["causal_relationship"] = torch.stack(cause_pairs) + mask

        batch["emotion"] = torch.nn.utils.rnn.pad_sequence(
            [i["emotion"] for i in instances],
            batch_first=True,
            padding_value=-1,
        )

        if "text" in self.modalities:
            batch["text"] = torch.nn.utils.rnn.pad_sequence(
                [i["text"] for i in instances],
                batch_first=True,
                padding_value=self.tokenizer.pad_token_id,
            )
        if "video" in self.modalities:
            batch["video"] = torch.nn.utils.rnn.pad_sequence(
                [i["video"] for i in instances],
                batch_first=True,
                padding_value=0,
            )
        if "audio" in self.modalities:
            batch["audio"] = torch.nn.utils.rnn.pad_sequence(
                [i["audio"] for i in instances],
                batch_first=True,
                padding_value=0,
            )

        return batch


if __name__ == "__main__":
    train_dataset = EmotionCausalDataset(split="train")
    val_dataset = EmotionCausalDataset(split="test")
    train_loader = DataLoader(
        train_dataset,
        batch_size=3,
        shuffle=True,
        num_workers=4,
        collate_fn=train_dataset.collater,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=3,
        num_workers=4,
        collate_fn=val_dataset.collater,
    )
