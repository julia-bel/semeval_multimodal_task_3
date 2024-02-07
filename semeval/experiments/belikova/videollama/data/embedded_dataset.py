import os
import torch
from torch.utils.data import DataLoader, Dataset


class EmbeddedDataset(Dataset):
    def __init__(
        self,
        root="/semeval/data/prompt_data",
        split="train",
    ):
        self.split = split
        self.root = root  # + "/" + split
        self.len = len(os.listdir(self.root))

    def __len__(self):
        return self.len

    def __getitem__(self, index, device="cpu"):
        file_path = os.path.join(self.root, f"embedded_batch_{index}.pt")
        result = torch.load(file_path, map_location=device)
        return result

    def collater(self, instances):
        return instances[0]


if __name__ == "__main__":
    train_dataset = EmbeddedDataset(split="train")
    val_dataset = EmbeddedDataset(split="val")
    train_loader = DataLoader(
        train_dataset,
        batch_size=1,
        shuffle=True,
        num_workers=4,
        collate_fn=train_dataset.collater,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        num_workers=4,
        collate_fn=val_dataset.collater,
    )
