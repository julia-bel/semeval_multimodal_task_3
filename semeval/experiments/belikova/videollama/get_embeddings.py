import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "6"

import torch
from transformers import AutoModelForCausalLM
from torch.utils.data import DataLoader
from omegaconf import OmegaConf
from tqdm import tqdm

from semeval.experiments.belikova.videollama.models import (
    VideoLLAMABackbone,
    LlamaEmbedding,
)
from semeval.experiments.belikova.videollama.data import (
    JointDataset,
)


if __name__ == "__main__":
    # data preprocessing
    modalities = ["text", "video"]
    train_dataset = JointDataset(split="train", modalities=modalities)
    train_loader = DataLoader(
        train_dataset,
        batch_size=1,
        num_workers=4,
        collate_fn=train_dataset.collater,
    )

    llama_model = AutoModelForCausalLM.from_pretrained(
        "semeval/experiments/belikova/videollama/ckpt/llama-2-7b-chat-hf",
        torch_dtype="auto",
        device_map="sequential",
    )
    llama_model.eval()

    # embeddings settings
    embeddings = {}
    device = torch.device("cuda")
    config = OmegaConf.load("configs/backbone.yaml")
    video_backbone = VideoLLAMABackbone.from_config(config)
    video_backbone.to(device)
    video_backbone.eval()
    embeddings["video"] = video_backbone.encode_videoQformer
    # audio_embedding = video_backbone.encode_audioQformer

    embedding_model = LlamaEmbedding(
        embeddings,
        llama_model=llama_model,
    )
    embedding_model.eval()
    prompts = [
        "Describe the behavior of the speaker in this video in one word:"
        # "You are a helpful language and visual assistant. Describe what is happening in this video in one word: ",
        # "You are a helpful language and visual assistant. Describe what the speaker do in this video in one word: ",
        # "You are a helpful language and visual assistant. Describe the behavior of the speaker in this video in one word: ",
    ]
    embedded_prompts = []
    with torch.no_grad():
        for prompt in prompts:
            tokens = train_dataset.tokenizer(
                prompt,
                return_tensors="pt",
            )
            embedded_prompts.append(
                (
                    llama_model.model.embed_tokens(tokens.input_ids).to(device),
                    tokens.attention_mask.to(device),
                )
            )
    save_directory = "/semeval/data/prompt_data/train"
    with torch.no_grad():
        for i, batch in tqdm(enumerate(train_loader)):
            try:
                batch = {k: v.to("cuda") for k, v in batch.items()}
                embedded_batch = embedding_model(batch, embedded_prompts[0])
                file_path = os.path.join(save_directory, f"embedded_batch_{i}.pt")
                torch.save(embedded_batch, file_path)
            except:
                continue

    # validation
    val_dataset = JointDataset(split="val", modalities=modalities)
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        num_workers=4,
        collate_fn=val_dataset.collater,
    )
    save_directory = "/semeval/data/prompt_data/val"
    with torch.no_grad():
        for i, batch in tqdm(enumerate(val_loader)):
            try:
                batch = {k: v.to("cuda") for k, v in batch.items()}
                embedded_batch = embedding_model(batch, embedded_prompts[0])
                file_path = os.path.join(save_directory, f"embedded_batch_{i}.pt")
                torch.save(embedded_batch, file_path)
            except:
                continue
