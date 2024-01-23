import torch
import torchmetrics
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.rnn as rnn_utils
import pytorch_lightning as pl


class Attention(nn.Module):
    def __init__(self, input_dim, attention_dim):
        super().__init__()
        self.attention_weights = nn.Linear(input_dim, attention_dim)
        self.context_vector = nn.Linear(attention_dim, 1, bias=False)

    def forward(self, x):
        attention_scores = self.context_vector(torch.tanh(self.attention_weights(x)))
        attention_weights = F.softmax(attention_scores, dim=2)
        weighted_average = torch.sum(x * attention_weights, dim=2)
        return weighted_average


class CausalClassifier(pl.LightningModule):
    def __init__(
        self,
        embeddings,
        input_dim=5120,
        attention_dim=128,
        modality_embedding_dim=512,
        emotion_embedding_dim=7,
        lstm_hidden_dim=128,
    ):
        super().__init__()
        self.embeddings = embeddings
        self.modalities = embeddings.keys()

        self.projections = nn.ModuleList(
            [
                nn.Sequential(nn.Linear(input_dim, modality_embedding_dim), nn.ReLU())
                for _ in range(len(embeddings))
            ]
        )
        self.attention_layers = nn.ModuleList(
            [
                Attention(modality_embedding_dim, attention_dim)
                for _ in range(len(embeddings))
            ]
        )

        self.emotion_bilstm = nn.LSTM(
            modality_embedding_dim + emotion_embedding_dim,
            lstm_hidden_dim,
            batch_first=True,
            bidirectional=True,
        )
        self.causal_bilstm = nn.LSTM(
            modality_embedding_dim,
            lstm_hidden_dim,
            batch_first=True,
            bidirectional=True,
        )

        self.cause_classifier = nn.Sequential(
            nn.Linear(2 * lstm_hidden_dim + modality_embedding_dim, lstm_hidden_dim),
            nn.ReLU(),
            nn.Linear(lstm_hidden_dim, 1),
        )

        self.f1 = torchmetrics.F1Score(
            task="binary", mdmc_average="global", ignore_index=-1
        )
        self.accuracy = torchmetrics.Accuracy(
            task="binary", mdmc_average="global", ignore_index=-1
        )

    def forward(self, modality_embeddings, utterance_lengths, emotion_embeddings):
        projections = [
            attention(project(embedding.float()))
            for attention, project, embedding in zip(
                self.attentions, self.projections, modality_embeddings
            )
        ]
        utterance_embeddings = torch.cat(projections, dim=2)

        causal_packed_utterances = rnn_utils.pack_padded_sequence(
            utterance_embeddings,
            utterance_lengths,
            batch_first=True,
            enforce_sorted=False,
        )
        causal_packed_output, _ = self.causal_bilstm(causal_packed_utterances)
        causal_utterances, _ = rnn_utils.pad_packed_sequence(
            causal_packed_output, batch_first=True
        )

        emotion_packed_utterances = rnn_utils.pack_padded_sequence(
            torch.cat((utterance_embeddings, emotion_embeddings), dim=2),
            utterance_lengths,
            batch_first=True,
            enforce_sorted=False,
        )
        emotion_packed_output, _ = self.causal_bilstm(emotion_packed_utterances)
        emotion_utterances, _ = rnn_utils.pad_packed_sequence(
            emotion_packed_output, batch_first=True
        )

        logits = torch.bmm(emotion_utterances, causal_utterances.transpose(1, 2))

        return logits

    def training_step(self, batch, batch_idx):
        utterance_lengths = batch["utterance_length"]
        emotion_embeddings = batch["emotion_embedding"]
        labels = batch["label"]

        assert all([m in batch for m in self.modalities]), "incorrect modality input"
        batch_size, num_utterances = batch[self.modalities[0]].shape[:2]
        modality_embeddings = [
            self.embeddings[m](
                batch[m].reshape(batch_size * num_utterances, *batch[m].shape[2:])
            )
            for m in self.modalities
        ]
        modality_embeddings = [
            e.reshape(batch_size, num_utterances, *e.shape[1:])
            for e in modality_embeddings
        ]
        logits = self(modality_embeddings, utterance_lengths, emotion_embeddings)

        range_tensor = torch.arange(num_utterances).expand(batch_size, num_utterances)
        mask = (range_tensor < utterance_lengths.unsqueeze(1)).float()
        loss = F.binary_cross_entropy_with_logits(logits, labels.float(), weight=mask)
        self.log("train_loss", loss, on_epoch=True)

        # Calculate metrics
        preds = (torch.sigmoid(logits) >= 0.5).float()
        self.log("train_f1", self.f1(preds, labels))
        self.log("train_accuracy", self.accuracy(preds, labels))

        return loss

    def validation_step(self, batch, batch_idx):
        utterance_lengths = batch["utterance_length"]
        emotion_embeddings = batch["emotion_embedding"]
        labels = batch["label"]

        assert all([m in batch for m in self.modalities]), "incorrect modality input"
        batch_size, num_utterances = batch[self.modalities[0]].shape[:2]
        modality_embeddings = [
            self.embeddings[m](
                batch[m].reshape(batch_size * num_utterances, *batch[m].shape[2:])
            )
            for m in self.modalities
        ]
        modality_embeddings = [
            e.reshape(batch_size, num_utterances, *e.shape[1:])
            for e in modality_embeddings
        ]
        logits = self(modality_embeddings, utterance_lengths, emotion_embeddings)

        range_tensor = torch.arange(num_utterances).expand(batch_size, num_utterances)
        mask = (range_tensor < utterance_lengths.unsqueeze(1)).float()
        loss = F.binary_cross_entropy_with_logits(logits, labels.float(), weight=mask)
        self.log("val_loss", loss, on_epoch=True)

        # Calculate metrics
        preds = (torch.sigmoid(logits) >= 0.5).float()
        self.log("val_f1", self.f1(preds, labels))
        self.log("val_accuracy", self.accuracy(preds, labels))

        return loss

    def test_step(self, batch, batch_idx):
        utterance_lengths = batch["utterance_length"]
        emotion_embeddings = batch["emotion_embedding"]
        labels = batch["label"]

        assert all([m in batch for m in self.modalities]), "incorrect modality input"
        batch_size, num_utterances = batch[self.modalities[0]].shape[:2]
        modality_embeddings = [
            self.embeddings[m](
                batch[m].reshape(batch_size * num_utterances, *batch[m].shape[2:])
            )
            for m in self.modalities
        ]
        modality_embeddings = [
            e.reshape(batch_size, num_utterances, *e.shape[1:])
            for e in modality_embeddings
        ]
        logits = self(modality_embeddings, utterance_lengths, emotion_embeddings)

        # Calculate metrics
        preds = (torch.sigmoid(logits) >= 0.5).float()
        self.log("test_f1", self.f1(preds, labels))
        self.log("test_accuracy", self.accuracy(preds, labels))

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
