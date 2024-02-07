import torch
import pytorch_lightning as pl
import torch.nn.functional as F
import torch.nn as nn
import torchmetrics


class EmbeddedCausalClassifier(pl.LightningModule):
    def __init__(
        self,
        modalities,
        num_classes=7,
        input_dim=4096,
        attention_dim=512,
        modality_embedding_dim=1024,
    ):
        super().__init__()
        self.num_classes = num_classes
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

        self.causal_bilstm = nn.LSTM(
            modality_embedding_dim * len(self.modalities),
            attention_dim // 2,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )

        self.causal_classifier = nn.Sequential(
            nn.Linear(attention_dim * 2, attention_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(attention_dim // 2, 1),
        )

        # Metrics
        self.f1_causal = torchmetrics.F1Score(
            task="binary",
            multidim_average="global",
            ignore_index=-1,
        )
        self.accuracy_causal = torchmetrics.Accuracy(
            task="binary",
            multidim_average="global",
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
        causal_embeddings, _ = self.causal_bilstm(utterance_embeddings)

        batch_size, seq_len, _ = emotion_embeddings.shape
        emotion_utterances = emotion_embeddings.unsqueeze(2).expand(-1, -1, seq_len, -1)
        causal_utterances = causal_embeddings.unsqueeze(1).expand(-1, seq_len, -1, -1)
        combined_utterances = torch.cat((emotion_utterances, causal_utterances), dim=-1)
        causal_logits = self.causal_classifier(combined_utterances).view(
            batch_size, seq_len, seq_len
        )
        # causal_predictions = (torch.sigmoid(causal_logits) >= 0.37).float()
        return causal_logits

    def training_step(self, batch, batch_idx):
        assert all([m in batch for m in self.modalities]), "incorrect modality input"
        # utterance_length is a binary mask: (batch_size, max_utterance_num)
        utterance_lengths = batch["utterance_length"]
        causal_labels = batch["causal_relationship"]
        causal_logits = self([batch[m] for m in self.modalities])

        weight = utterance_lengths[:, :, None] @ utterance_lengths[:, None, :]
        causal_loss = F.binary_cross_entropy_with_logits(
            causal_logits, causal_labels.float(), weight=weight + (causal_labels * 3)
        )
        causal_predictions = (torch.sigmoid(causal_logits) >= 0.5).float()

        # Loss and Metrics

        self.log("train_loss", causal_loss, on_step=False, on_epoch=True)

        self.log(
            "train_causal_f1",
            self.f1_causal(causal_predictions, causal_labels),
            on_step=False,
            on_epoch=True,
        )
        self.log(
            "train_causal_accuracy",
            self.accuracy_causal(causal_predictions, causal_labels),
            on_step=False,
            on_epoch=True,
        )
        return causal_loss

    def validation_step(self, batch, batch_idx):
        assert all([m in batch for m in self.modalities]), "incorrect modality input"
        # utterance_length is a binary mask: (batch_size, max_utterance_num)
        utterance_lengths = batch["utterance_length"]
        causal_labels = batch["causal_relationship"]
        causal_logits = self([batch[m] for m in self.modalities])

        weight = utterance_lengths[:, :, None] @ utterance_lengths[:, None, :]
        causal_loss = F.binary_cross_entropy_with_logits(
            causal_logits, causal_labels.float(), weight=weight + (causal_labels * 3)
        )
        causal_predictions = (torch.sigmoid(causal_logits) >= 0.5).float()

        # Loss and Metrics

        self.log("val_loss", causal_loss, on_step=False, on_epoch=True)

        self.log(
            "val_causal_f1",
            self.f1_causal(causal_predictions, causal_labels),
            on_step=False,
            on_epoch=True,
        )
        self.log(
            "val_causal_accuracy",
            self.accuracy_causal(causal_predictions, causal_labels),
            on_step=False,
            on_epoch=True,
        )
        return causal_loss

    def on_train_epoch_start(self):
        self.f1_causal.reset()
        self.accuracy_causal.reset()

    def on_validation_epoch_start(self):
        self.f1_causal.reset()
        self.accuracy_causal.reset()

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
