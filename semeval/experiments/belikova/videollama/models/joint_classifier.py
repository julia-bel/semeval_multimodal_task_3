import torch
import torch.nn as nn
import pytorch_lightning as pl
import torch.nn.functional as F
import torchmetrics


class EmotionCausalClassifier(pl.LightningModule):
    def __init__(
        self,
        embeddings,
        emotion_classifier,
        causal_classifier,
        input_dim=5120,
        hidden_dim=512,
        num_classes=7,
    ):
        super().__init__()
        # Backbone
        self.modalities = embeddings.keys()
        self.embeddings = embeddings
        self.projections = nn.ModuleList(
            [nn.Linear(input_dim, hidden_dim) for _ in range(len(embeddings))]
        )

        # Heads
        self.emotion_classifier = emotion_classifier
        self.causal_classifier = causal_classifier

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
    def forward(self, modalities, utterance_lengths):
        """
        Args:
            modalities (dict): {
                "text": (batch_size, max_utterance_num, max_text_len),
                "video": (batch_size, max_utterance_num, channels, frames, height, width),
                "audio": (batch_size, max_utterance_num, frames, channels, height, width)
            }
            utterance_lengths (torch.tensor): (batch_size, max_utterance_num) mask of padding in utterance dim
        Returns:
            tuple: emotion and causal predictions
        """
        batch_size, num_utterances = utterance_lengths.shape

        # Emotion classification
        utt_modality_embeddings = [
            self.embeddings[m](
                modalities[m].reshape(
                    batch_size * num_utterances, *modalities[m].shape[2:]
                )
            )
            for m in self.modalities
        ]
        emotion_logits = self.emotion_classifier(utt_modality_embeddings)
        emotion_predictions = torch.argmax(emotion_logits, dim=1)

        # Causal classification
        conv_modality_embeddings = [
            e.reshape(batch_size, num_utterances, *e.shape[1:])
            for e in utt_modality_embeddings
        ]
        causal_logits = self.causal_classifier(
            conv_modality_embeddings,
            torch.sum(utterance_lengths, dim=1),
            emotion_logits.view(batch_size, num_utterances, -1),
        )
        causal_predictions = (torch.sigmoid(causal_logits) >= 0.5).float()
        return emotion_predictions, causal_predictions

    def training_step(self, batch, batch_idx):
        assert all([m in batch for m in self.modalities]), "incorrect modality input"
        # utterance_length is a binary mask: (batch_size, max_utterance_num)
        utterance_lengths = batch["utterance_length"]
        causal_labels = batch["causal_relationship"]
        emotion_labels = batch["emotion"].view(-1)
        batch_size, num_utterances = utterance_lengths.shape

        # Emotion classification
        utt_modality_embeddings = [
            self.embeddings[m](
                batch[m].reshape(batch_size * num_utterances, *batch[m].shape[2:])
            )
            for m in self.modalities
        ]
        emotion_logits = self.emotion_classifier(utt_modality_embeddings)
        emotion_loss = F.cross_entropy(emotion_logits, emotion_labels, ignore_index=-1)
        emotion_predictions = torch.argmax(emotion_logits, dim=1)

        # Causal classification
        conv_modality_embeddings = [
            e.reshape(batch_size, num_utterances, *e.shape[1:])
            for e in utt_modality_embeddings
        ]
        causal_logits = self.causal_classifier(
            conv_modality_embeddings,
            torch.sum(utterance_lengths, dim=1),
            emotion_logits.view(batch_size, num_utterances, -1),
        )
        weight = utterance_lengths[:, :, None] @ utterance_lengths[:, None, :]
        causal_loss = F.binary_cross_entropy_with_logits(
            causal_logits, causal_labels.float(), weight=weight
        )
        causal_predictions = (torch.sigmoid(causal_logits) >= 0.5).float()

        # Loss and Metrics
        combined_loss = emotion_loss + causal_loss
        self.log("train_loss", combined_loss, on_epoch=True)

        self.log(
            "train_emotion_f1",
            self.f1_emotion(emotion_predictions, emotion_labels),
        )
        self.log(
            "train_emotion_accuracy",
            self.accuracy_emotion(emotion_predictions, emotion_labels),
        )

        self.log(
            "train_causal_f1",
            self.f1_causal(causal_predictions, causal_labels),
        )
        self.log(
            "train_causal_accuracy",
            self.accuracy_causal(causal_predictions, causal_labels),
        )
        return combined_loss

    def validation_step(self, batch, batch_idx):
        assert all([m in batch for m in self.modalities]), "incorrect modality input"
        # utterance_length is a binary mask: (batch_size, max_utterance_num)
        utterance_lengths = batch["utterance_length"]
        causal_labels = batch["causal_relationship"]
        emotion_labels = batch["emotion"].view(-1)

        batch_size, num_utterances = utterance_lengths.shape
        emotion_labels = emotion_labels.reshape(
            batch_size * num_utterances, *emotion_labels.shape[2:]
        )

        # Emotion classification
        utt_modality_embeddings = [
            self.embeddings[m](
                batch[m].reshape(batch_size * num_utterances, *batch[m].shape[2:])
            )
            for m in self.modalities
        ]
        emotion_logits = self.emotion_classifier(utt_modality_embeddings)
        emotion_loss = F.cross_entropy(emotion_logits, emotion_labels, ignore_index=-1)
        emotion_predictions = torch.argmax(emotion_logits, dim=1)

        # Causal classification
        conv_modality_embeddings = [
            e.reshape(batch_size, num_utterances, *e.shape[1:])
            for e in utt_modality_embeddings
        ]
        causal_logits = self.causal_classifier(
            conv_modality_embeddings,
            torch.sum(utterance_lengths, dim=1),
            emotion_logits.view(batch_size, num_utterances, -1),
        )
        weight = utterance_lengths[:, :, None] @ utterance_lengths[:, None, :]
        causal_loss = F.binary_cross_entropy_with_logits(
            causal_logits, causal_labels.float(), weight=weight
        )
        causal_predictions = (torch.sigmoid(causal_logits) >= 0.5).float()

        # Loss and Metrics
        combined_loss = emotion_loss + causal_loss
        self.log("val_loss", combined_loss, on_epoch=True)

        self.log(
            "val_emotion_f1",
            self.f1_emotion(emotion_predictions, emotion_labels),
        )
        self.log(
            "val_emotion_accuracy",
            self.accuracy_emotion(emotion_predictions, emotion_labels),
        )

        self.log(
            "val_causal_f1",
            self.f1_causal(causal_predictions, causal_labels),
        )
        self.log(
            "val_causal_accuracy",
            self.accuracy_causal(causal_predictions, causal_labels),
        )
        return combined_loss

    def test_step(self, batch, batch_idx):
        assert all([m in batch for m in self.modalities]), "incorrect modality input"
        # utterance_length is a binary mask: (batch_size, max_utterance_num)
        utterance_lengths = batch["utterance_length"]
        causal_labels = batch["causal_relationship"]
        emotion_labels = batch["emotion"].view(-1)
        batch_size, num_utterances = utterance_lengths.shape

        # Emotion classification
        utt_modality_embeddings = [
            self.embeddings[m](
                batch[m].reshape(batch_size * num_utterances, *batch[m].shape[2:])
            )
            for m in self.modalities
        ]
        emotion_logits = self.emotion_classifier(utt_modality_embeddings)
        emotion_predictions = torch.argmax(emotion_logits, dim=1)

        # Causal classification
        conv_modality_embeddings = [
            e.reshape(batch_size, num_utterances, *e.shape[1:])
            for e in utt_modality_embeddings
        ]
        causal_logits = self.causal_classifier(
            conv_modality_embeddings,
            torch.sum(utterance_lengths, dim=1),
            emotion_logits.view(batch_size, num_utterances, -1),
        )
        causal_predictions = (torch.sigmoid(causal_logits) >= 0.5).float()

        # Metrics
        self.log(
            "test_emotion_f1",
            self.f1_emotion(emotion_predictions, emotion_labels),
        )
        self.log(
            "test_emotion_accuracy",
            self.accuracy_emotion(emotion_predictions, emotion_labels),
        )

        self.log(
            "test_causal_f1",
            self.f1_causal(causal_predictions, causal_labels),
        )
        self.log(
            "test_causal_accuracy",
            self.accuracy_causal(causal_predictions, causal_labels),
        )

    def on_train_epoch_start(self):
        self.f1_causal.reset()
        self.f1_emotion.reset()
        self.accuracy_causal.reset()
        self.accuracy_emotion.reset()

    def on_validation_epoch_start(self):
        self.f1_causal.reset()
        self.f1_emotion.reset()
        self.accuracy_causal.reset()
        self.accuracy_emotion.reset()

    def on_test_epoch_start(self):
        self.f1_causal.reset()
        self.f1_emotion.reset()
        self.accuracy_causal.reset()
        self.accuracy_emotion.reset()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        return optimizer
