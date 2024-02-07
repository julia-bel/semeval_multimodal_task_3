import torch
import torch.nn as nn


class LlamaEmbedding(nn.Module):
    def __init__(
        self,
        embeddings,
        llama_model,
    ):
        super().__init__()
        self.embeddings = embeddings
        self.llama_model = llama_model

    def llama_embedding(self, batch, attention_mask):
        if batch.dim() > 2:
            last_hidden_state = self.llama_model(
                inputs_embeds=batch, output_hidden_states=True, return_dict=True
            ).hidden_states[-1]
        else:
            last_hidden_state = self.llama_model(
                input_ids=batch, output_hidden_states=True, return_dict=True
            ).hidden_states[-1]
        idx_of_the_last_non_padding_token = attention_mask.bool().sum(1) - 1
        embedding = last_hidden_state[
            torch.arange(last_hidden_state.shape[0]), idx_of_the_last_non_padding_token
        ]
        return embedding

    def forward(self, batch, prompt):
        # batch = {k: old_batch[k].clone() for k in old_batch}
        batch_size, num_utterances = batch["text"].shape[:2]
        modality_embeddings = {}

        if "text" in batch:
            modality_embeddings["text"] = self.llama_embedding(
                batch["text"][:, :, 0, :].view(batch_size * num_utterances, -1),
                batch["text"][:, :, 1, :].view(batch_size * num_utterances, -1),
            )
        for modality in ["video", "audio"]:
            if modality in batch:
                embed, mask = self.embeddings[modality](
                    batch[modality].view(
                        batch_size * num_utterances, *batch[modality].shape[2:]
                    )
                )
                modality_embeddings[modality] = self.llama_embedding(
                    torch.cat(
                        (prompt[0].repeat((embed.size(0), 1, 1)), embed), dim=1
                    ),  # embed
                    torch.cat(
                        (prompt[1].repeat((mask.size(0), 1)), mask), dim=1
                    ),  # mask
                )
        for m, value in modality_embeddings.items():
            batch[m] = value.view(batch_size, num_utterances, *value.shape[1:])

        return batch
