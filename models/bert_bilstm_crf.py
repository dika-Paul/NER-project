import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torchcrf import CRF
from transformers import AutoConfig, AutoModel


class BertBiLstmCrfNER(nn.Module):
    """
    Tokenizer -> BERT -> BiLSTM -> Linear -> CRF model for word-level NER.

    The tokenizer is intentionally kept in the data pipeline. This module receives
    tokenized BERT inputs plus first-subword positions, then produces CRF emissions
    aligned to the original word sequence.
    """

    def __init__(
        self,
        model_name,
        num_labels,
        lstm_hidden_size=256,
        lstm_num_layers=1,
        dropout=0.25,
        id2label=None,
        label2id=None,
        expected_bert_hidden_size=768,
    ):
        super().__init__()

        self.config = AutoConfig.from_pretrained(
            model_name,
            num_labels=num_labels,
            id2label=id2label,
            label2id=label2id,
        )

        if (
            expected_bert_hidden_size is not None
            and self.config.hidden_size != expected_bert_hidden_size
        ):
            raise ValueError(
                f"{model_name} hidden size is {self.config.hidden_size}, "
                f"expected {expected_bert_hidden_size}."
            )

        self.bert = AutoModel.from_pretrained(model_name, config=self.config)

        self.bilstm = nn.LSTM(
            input_size=self.config.hidden_size,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if lstm_num_layers > 1 else 0.0,
        )

        self.bert_hidden_size = self.config.hidden_size
        self.lstm_hidden_size = lstm_hidden_size
        self.bilstm_output_size = lstm_hidden_size * 2

        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.bilstm_output_size, num_labels)
        self.crf = CRF(num_labels, batch_first=True)

    def _gather_first_subword_states(self, sequence_output, first_subword_positions):
        """
        Convert subword-level BERT states to word-level states by taking the first
        subword state for each original token.
        """
        hidden_size = sequence_output.size(-1)
        gather_index = first_subword_positions.unsqueeze(-1).expand(-1, -1, hidden_size)
        return torch.gather(sequence_output, dim=1, index=gather_index)

    def _encode_lstm_branch(self, bert_word_features, word_attention_mask):
        lengths = word_attention_mask.sum(dim=1).to(dtype=torch.long)

        packed = pack_padded_sequence(
            bert_word_features,
            lengths.cpu(),
            batch_first=True,
            enforce_sorted=False,
        )
        packed_output, _ = self.bilstm(packed)

        lstm_output, _ = pad_packed_sequence(
            packed_output,
            batch_first=True,
            total_length=bert_word_features.size(1),
        )
        return lstm_output

    def _get_emissions(
        self,
        input_ids,
        attention_mask,
        word_attention_mask,
        first_subword_positions,
        token_type_ids=None,
    ):
        bert_outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        sequence_output = bert_outputs.last_hidden_state
        bert_word_features = self._gather_first_subword_states(
            sequence_output, first_subword_positions
        )

        lstm_features = self._encode_lstm_branch(
            bert_word_features, word_attention_mask
        )
        lstm_features = self.dropout(lstm_features)

        emissions = self.classifier(lstm_features)
        return emissions

    def neg_log_likelihood(
        self,
        input_ids,
        attention_mask,
        word_attention_mask,
        first_subword_positions,
        labels,
        token_type_ids=None,
    ):
        mask = word_attention_mask.bool()
        emissions = self._get_emissions(
            input_ids=input_ids,
            attention_mask=attention_mask,
            word_attention_mask=word_attention_mask,
            first_subword_positions=first_subword_positions,
            token_type_ids=token_type_ids,
        )
        return -self.crf(emissions, labels, mask=mask, reduction="mean")

    def forward(
        self,
        input_ids,
        attention_mask,
        word_attention_mask,
        first_subword_positions,
        labels=None,
        token_type_ids=None,
    ):
        mask = word_attention_mask.bool()
        emissions = self._get_emissions(
            input_ids=input_ids,
            attention_mask=attention_mask,
            word_attention_mask=word_attention_mask,
            first_subword_positions=first_subword_positions,
            token_type_ids=token_type_ids,
        )

        output_dict = {
            "emissions": emissions,
            "predictions": self.crf.decode(emissions, mask=mask),
        }

        if labels is not None:
            output_dict["loss"] = -self.crf(
                emissions, labels, mask=mask, reduction="mean"
            )

        return output_dict
