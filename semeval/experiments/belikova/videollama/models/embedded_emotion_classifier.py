import torch
import pytorch_lightning as pl
import torch.nn.functional as F
import torch.nn as nn
import torchmetrics


class EmbeddedEmotionClassifier(pl.LightningModule):
    def __init__(
        self,
        modalities,
        input_dim=4096,
        attention_dim=512,
        modality_embedding_dim=1024,
        num_classes=7,
    ):
        super().__init__()
        self.modalities = modalities

        self.projections = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(input_dim, modality_embedding_dim * 2),
                    nn.ReLU(),
                    nn.Dropout(0.3),
                    nn.Linear(modality_embedding_dim * 2, modality_embedding_dim),
                    nn.ReLU(),
                    nn.Dropout(0.3),
                )
                for _ in range(len(modalities))
            ]
        )

        self.emotion_linear = nn.Sequential(
            nn.Linear(modality_embedding_dim * len(self.modalities), attention_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
        )

        self.emotion_classifier = nn.Sequential(
            nn.Linear(attention_dim, attention_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(attention_dim // 2, num_classes),
        )

        # Metrics
        self.f1_emotion = torchmetrics.F1Score(
            task="multiclass",
            num_classes=num_classes,
            average="macro",
            ignore_index=-1,
        )
        self.accuracy_emotion = torchmetrics.Accuracy(
            task="multiclass",
            num_classes=num_classes,
            ignore_index=-1,
        )

    # Inference only
    def forward(self, modality_embeddings):
        """
        Args:
            modalities (dict): {
                "text": (batch_size, max_utterance_num, 2, max_text_len),
                "video": (batch_size, max_utterance_num, channels, frames, height, width),
                "audio": (batch_size, max_utterance_num, frames, channels, height, width)
            }
            utterance_lengths (torch.tensor): (batch_size, max_utterance_num) mask of padding in utterance dim
        Returns:
            tuple: emotion and causal predictions
        """
        projections = [
            project(embedding.float())
            for project, embedding in zip(self.projections, modality_embeddings)
        ]

        utterance_embeddings = torch.cat(projections, dim=2)
        emotion_embeddings = self.emotion_linear(utterance_embeddings)
        emotion_logits = self.emotion_classifier(emotion_embeddings)
        # emotion_predictions = torch.argmax(emotion_logits, dim=2)
        # emotion_logits = torch.nn.Softmax(dim=2)(emotion_logits)
        return emotion_logits

    def training_step(self, batch, batch_idx):
        assert all([m in batch for m in self.modalities]), "incorrect modality input"
        emotion_labels = batch["emotion"]

        emotion_logits = self([batch[m] for m in self.modalities])
        emotion_loss = F.cross_entropy(
            emotion_logits.view(-1, self.num_classes),
            emotion_labels.view(-1),
            ignore_index=-1,
        )
        emotion_predictions = torch.argmax(emotion_logits, dim=2)

        # Loss and Metrics
        self.log("train_loss", emotion_loss, on_step=False, on_epoch=True)

        self.log(
            "train_emotion_f1",
            self.f1_emotion(emotion_predictions, emotion_labels),
            on_step=False,
            on_epoch=True,
        )
        self.log(
            "train_emotion_accuracy",
            self.accuracy_emotion(emotion_predictions, emotion_labels),
            on_step=False,
            on_epoch=True,
        )
        return emotion_loss

    def validation_step(self, batch, batch_idx):
        assert all([m in batch for m in self.modalities]), "incorrect modality input"
        emotion_labels = batch["emotion"]

        emotion_logits = self([batch[m] for m in self.modalities])
        emotion_loss = F.cross_entropy(
            emotion_logits.view(-1, self.num_classes),
            emotion_labels.view(-1),
            ignore_index=-1,
        )
        emotion_predictions = torch.argmax(emotion_logits, dim=2)

        # Loss and Metrics
        self.log("val_loss", emotion_loss, on_step=False, on_epoch=True)

        self.log(
            "val_emotion_f1",
            self.f1_emotion(emotion_predictions, emotion_labels),
            on_step=False,
            on_epoch=True,
        )
        self.log(
            "val_emotion_accuracy",
            self.accuracy_emotion(emotion_predictions, emotion_labels),
            on_step=False,
            on_epoch=True,
        )

        return emotion_loss

    def on_train_epoch_start(self):
        self.f1_emotion.reset()
        self.accuracy_emotion.reset()

    def on_validation_epoch_start(self):
        self.f1_emotion.reset()
        self.accuracy_emotion.reset()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        lr_scheduler = {
            "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="min", factor=0.1, patience=10
            ),
            "monitor": "val_loss",
            "interval": "epoch",
            "frequency": 1,
        }
        return [optimizer], [lr_scheduler]
