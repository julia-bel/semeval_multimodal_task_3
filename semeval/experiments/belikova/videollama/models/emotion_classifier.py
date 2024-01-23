import torch
import torchmetrics
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl


class Attention(nn.Module):
    def __init__(self, input_dim, attention_dim):
        super().__init__()
        self.attention_weights = nn.Linear(input_dim, attention_dim)
        self.context_vector = nn.Linear(attention_dim, 1, bias=False)

    def forward(self, x):
        attention_scores = self.context_vector(torch.tanh(self.attention_weights(x)))
        attention_weights = F.softmax(attention_scores, dim=1)
        weighted_average = torch.sum(x * attention_weights, dim=1)
        return weighted_average


class EmotionClassifier(pl.LightningModule):
    def __init__(
        self,
        embeddings,
        input_dim=5120,
        hidden_dim=512,
        attention_dim=128,
        num_classes=7,
    ):
        super().__init__()
        self.embeddings = embeddings
        self.modalities = embeddings.keys()

        self.projections = nn.ModuleList(
            [
                nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU())
                for _ in range(len(embeddings))
            ]
        )
        self.attentions = nn.ModuleList(
            [Attention(hidden_dim, attention_dim) for _ in range(len(embeddings))]
        )

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, num_classes),
        )

        self.accuracy = torchmetrics.Accuracy(
            task="multiclass", num_classes=num_classes
        )
        self.f1 = torchmetrics.F1Score(
            task="multiclass", num_classes=num_classes, average="macro"
        )

    def forward(self, embeddings):
        projections = [
            attention(project(embedding.float()))
            for attention, project, embedding in zip(
                self.attentions, self.projections, embeddings
            )
        ]
        concat_features = torch.cat(projections, dim=1)
        logits = self.classifier(concat_features)
        return logits

    def training_step(self, batch, batch_idx):
        embeddings = [self.embeddings[mod](batch[mod]) for mod in self.modalities]
        logits = self(embeddings)
        labels = batch["label"]

        loss = F.cross_entropy(logits, labels)
        self.log("train_loss", loss)

        preds = torch.argmax(logits, dim=1)
        self.log(
            "train_f1",
            self.f1(preds, labels),
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        self.log(
            "train_acc",
            self.accuracy(preds, labels),
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        return loss

    def validation_step(self, batch, batch_idx):
        embeddings = [self.embeddings[mod](batch[mod]) for mod in self.modalities]
        logits = self(embeddings)
        labels = batch["label"]

        loss = F.cross_entropy(logits, labels)
        self.log("val_loss", loss)

        preds = torch.argmax(logits, dim=1)
        self.log(
            "val_f1",
            self.f1(preds, labels),
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        self.log(
            "val_acc",
            self.accuracy(preds, labels),
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )

        return loss

    def test_step(self, batch, batch_idx):
        embeddings = [self.embeddings[mod](batch[mod]) for mod in self.modalities]
        logits = self(embeddings)
        labels = batch["label"]

        preds = torch.argmax(logits, dim=1)
        self.log(
            "test_f1",
            self.f1(preds, labels),
            on_step=False,
            on_epoch=True,
        )
        self.log(
            "test_acc",
            self.accuracy(preds, labels),
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )

    def on_train_epoch_start(self):
        self.f1.reset()
        self.accuracy.reset()

    def on_validation_epoch_start(self):
        self.f1.reset()
        self.accuracy.reset()

    def on_test_epoch_start(self):
        self.f1.reset()
        self.accuracy.reset()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer
