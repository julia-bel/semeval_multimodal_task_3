import os
import json

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "6"

import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from datasets import load_dataset

from semeval.experiments.belikova.videollama.models import (
    EmbeddedEmotionClassifier,
    EmbeddedCausalClassifier,
)
from semeval.experiments.belikova.videollama.data import (
    EmbeddedDataset,
)

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


def process_prediction(emotions, logits, causal_pairs):
    emotions = [labels2emotions[e] for e in emotions.tolist()]
    pairs = torch.nonzero(causal_pairs, as_tuple=False).tolist()
    result = []
    used_emotions = set()
    for p in pairs:
        if emotions[p[0]] != "neutral":
            result.append([str(p[0] + 1) + "_" + emotions[p[0]], str(p[1] + 1)])
        used_emotions.add(p[0])
    for i, emotion in enumerate(emotions):
        if (
            emotion != "neutral"
            and i not in used_emotions
            and torch.max(logits[i]) >= 0.8
        ):
            if i > 0:
                result.append([str(i + 1) + "_" + emotion, str(i)])
            else:
                result.append([str(i + 1) + "_" + emotion, str(i + 1)])
    return result


if __name__ == "__main__":
    # data preprocessing
    device = torch.device("cuda")
    modalities = ["text", "video"]
    val_dataset = EmbeddedDataset(split="val", root="semeval/data/test/basic_data")
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        num_workers=4,
        collate_fn=val_dataset.collater,
    )

    # models settings
    emotion_model = EmbeddedEmotionClassifier.load_from_checkpoint(
        checkpoint_path="semeval/models/emotion_best/epoch=58.ckpt",
        modalities=modalities,
    )
    emotion_model.eval()
    causal_model = EmbeddedCausalClassifier.load_from_checkpoint(
        checkpoint_path="semeval/models/causal_best/epoch=69.ckpt",
        modalities=modalities,
    )
    causal_model.eval()

    file = "semeval/data/test/Subtask_2_test.json"
    dataset = load_dataset("json", data_files=file, split="train")
    new_data = []
    with torch.no_grad():
        for i, batch in tqdm(enumerate(val_loader)):
            batch = {k: v.to("cuda") for k, v in batch.items()}

            emotion_logits = emotion_model(batch)
            emotion_predictions = torch.argmax(emotion_logits, dim=2).cpu()
            emotion_logits = torch.nn.Softmax(dim=2)(emotion_logits).cpu().float()

            causal_logits = causal_model(batch)
            causal_predictions = (torch.sigmoid(causal_logits) >= 0.37).cpu().float()

            new_conv = dataset[i].copy()
            new_conv["emotion-cause_pairs"] = process_prediction(
                emotion_predictions[0], emotion_logits[0], causal_predictions[0]
            )
            new_data.append(new_conv)

    with open("semeval/data/test/Subtask_2_pred.json", "w") as f:
        json.dump(new_data, f)
