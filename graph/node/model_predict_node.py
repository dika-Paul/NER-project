from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer

from graph.graph_state import GraphState
from models.bert_bilstm_crf import BertBiLstmCrfNER
from models.bert_softmax import BertSoftmaxNER
from models.bilstm_crf import BiLSTM_CRF
from models.matscibert_softmax import MatSciBertSoftmaxNER
from utils.bert_bilstm_crf.data_utils import build_collate_fn as build_parallel_collate_fn


PREDICT_BATCH_SIZE = 16
PREDICT_NUM_WORKERS = 0


class _TokenPredictionDataset(Dataset):
    def __init__(self, samples: list[tuple[str, list[str]]]):
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[str, list[str]]:
        return self.samples[index]


def _load_checkpoint(best_model_path: str) -> tuple[dict[str, Any], Path]:
    if not best_model_path:
        raise ValueError("best_model_path cannot be empty.")

    path = Path(best_model_path).expanduser()
    if not path.exists() or not path.is_file():
        raise FileNotFoundError(f"best_model_path does not exist or is not a file: {path}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint = torch.load(path, map_location=device)
    if not isinstance(checkpoint, dict):
        raise ValueError("checkpoint must be a dict.")

    return checkpoint, path


def _idx2tag_from_checkpoint(checkpoint: dict[str, Any]) -> dict[int, str]:
    if "tag2idx" not in checkpoint:
        raise ValueError("checkpoint must contain tag2idx.")

    tag2idx = checkpoint["tag2idx"]
    if "O" not in tag2idx:
        raise ValueError("tag2idx must contain O.")

    idx2tag = checkpoint.get("idx2tag")
    if idx2tag is None:
        idx2tag = {idx: tag for tag, idx in tag2idx.items()}
    else:
        idx2tag = {int(idx): tag for idx, tag in idx2tag.items()}

    return idx2tag


def _validate_tokens(sample_id: str, sample: dict[str, Any]) -> list[str]:
    if "tokens" not in sample:
        raise ValueError(f"current_batch sample {sample_id} must contain tokens.")
    tokens = sample["tokens"]
    if not isinstance(tokens, list):
        raise ValueError(f"current_batch sample {sample_id} tokens must be a list.")
    return [str(token) for token in tokens]


def _prepare_prediction_samples(
    current_batch: dict[str, dict[str, Any]],
) -> tuple[dict[str, list[tuple[str, str]]], list[tuple[str, list[str]]]]:
    results = {}
    samples = []

    for sample_id, sample in current_batch.items():
        tokens = _validate_tokens(sample_id, sample)
        results[sample_id] = []
        if not tokens:
            continue
        samples.append((sample_id, tokens))

    return results, samples


def _prediction_loader(dataset: Dataset, collate_fn) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=PREDICT_BATCH_SIZE,
        shuffle=False,
        num_workers=PREDICT_NUM_WORKERS,
        collate_fn=collate_fn,
    )


def _checkpoint_model_type(checkpoint: dict[str, Any], checkpoint_path: Path) -> str:
    if "model_state_dict" in checkpoint:
        return "bilstm_crf"

    if "model" not in checkpoint:
        raise ValueError("checkpoint must contain model or model_state_dict.")

    if "word2idx" in checkpoint:
        return "bert_bilstm_crf"

    model_name = str(checkpoint.get("model_name", "bert-base-cased"))
    if model_name == "m3rg-iitd/matscibert" or "matscibert" in str(checkpoint_path).lower():
        return "matscibert_softmax"

    return "bert_softmax"


def _predict_bilstm_crf(
    checkpoint: dict[str, Any],
    current_batch: dict[str, dict[str, Any]],
) -> dict[str, list[tuple[str, str]]]:
    if "word2idx" not in checkpoint:
        raise ValueError("BiLSTM-CRF checkpoint must contain word2idx.")

    word2idx = checkpoint["word2idx"]
    tag2idx = checkpoint["tag2idx"]
    idx2tag = _idx2tag_from_checkpoint(checkpoint)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = BiLSTM_CRF(
        vocab_size=len(word2idx),
        tag_to_ix=tag2idx,
        embedding_dim=checkpoint.get("embedding_dim", 300),
        hidden_dim=checkpoint.get("hidden_dim", 384),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    results, prediction_samples = _prepare_prediction_samples(current_batch)
    if not prediction_samples:
        return results

    unk_id = word2idx.get("<UNK>", 1)

    def collate_fn(batch: list[tuple[str, list[str]]]) -> dict[str, Any]:
        sample_ids, token_sequences = zip(*batch)
        lengths = torch.tensor([len(tokens) for tokens in token_sequences], dtype=torch.long)
        max_length = int(lengths.max().item())
        input_ids = torch.zeros((len(batch), max_length), dtype=torch.long)

        for row_index, tokens in enumerate(token_sequences):
            token_ids = [word2idx.get(token, unk_id) for token in tokens]
            input_ids[row_index, : len(token_ids)] = torch.tensor(token_ids, dtype=torch.long)

        return {
            "sample_ids": list(sample_ids),
            "tokens": [list(tokens) for tokens in token_sequences],
            "input_ids": input_ids,
            "lengths": lengths,
        }

    dataloader = _prediction_loader(
        _TokenPredictionDataset(prediction_samples),
        collate_fn,
    )

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            lengths = batch["lengths"].to(device)
            batch_pred_ids = model(input_ids, lengths)

            for sample_id, tokens, pred_ids in zip(
                batch["sample_ids"],
                batch["tokens"],
                batch_pred_ids,
            ):
                labels = [idx2tag[int(pred_id)] for pred_id in pred_ids]
                results[sample_id] = list(zip(tokens[: len(labels)], labels))

    return results


def _predict_bert_softmax(
    checkpoint: dict[str, Any],
    current_batch: dict[str, dict[str, Any]],
    *,
    use_matscibert: bool,
) -> dict[str, list[tuple[str, str]]]:
    tag2idx = checkpoint["tag2idx"]
    idx2tag = _idx2tag_from_checkpoint(checkpoint)
    model_name = checkpoint.get(
        "model_name",
        "m3rg-iitd/matscibert" if use_matscibert else "bert-base-cased",
    )
    max_length = checkpoint.get("max_length", 128)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    model_cls = MatSciBertSoftmaxNER if use_matscibert else BertSoftmaxNER
    model = model_cls(
        model_name=model_name,
        num_labels=len(tag2idx),
        id2label=idx2tag,
        label2id=tag2idx,
    ).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    results, prediction_samples = _prepare_prediction_samples(current_batch)
    if not prediction_samples:
        return results

    def collate_fn(batch: list[tuple[str, list[str]]]) -> dict[str, Any]:
        sample_ids, token_sequences = zip(*batch)
        encodings = tokenizer(
            list(token_sequences),
            is_split_into_words=True,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )

        return {
            "sample_ids": list(sample_ids),
            "tokens": [list(tokens) for tokens in token_sequences],
            "word_ids": [
                encodings.word_ids(batch_index=index)
                for index in range(len(token_sequences))
            ],
            "input_ids": encodings["input_ids"],
            "attention_mask": encodings["attention_mask"],
            "token_type_ids": encodings.get("token_type_ids"),
        }

    dataloader = _prediction_loader(
        _TokenPredictionDataset(prediction_samples),
        collate_fn,
    )

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            token_type_ids = batch["token_type_ids"]
            if token_type_ids is not None:
                token_type_ids = token_type_ids.to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
            )
            batch_pred_ids = torch.argmax(outputs["logits"], dim=-1).cpu().tolist()

            for sample_id, tokens, word_ids, pred_ids in zip(
                batch["sample_ids"],
                batch["tokens"],
                batch["word_ids"],
                batch_pred_ids,
            ):
                token_label_pairs = []
                seen_word_ids = set()
                for token_position, word_id in enumerate(word_ids):
                    if word_id is None or word_id in seen_word_ids:
                        continue
                    seen_word_ids.add(word_id)
                    if word_id >= len(tokens):
                        continue

                    token_label_pairs.append(
                        (tokens[word_id], idx2tag[int(pred_ids[token_position])])
                    )

                results[sample_id] = token_label_pairs

    return results


def _predict_bert_bilstm_crf(
    checkpoint: dict[str, Any],
    current_batch: dict[str, dict[str, Any]],
) -> dict[str, list[tuple[str, str]]]:
    if "word2idx" not in checkpoint:
        raise ValueError("BERT-BiLSTM-CRF checkpoint must contain word2idx.")

    word2idx = checkpoint["word2idx"]
    tag2idx = checkpoint["tag2idx"]
    idx2tag = _idx2tag_from_checkpoint(checkpoint)
    model_name = checkpoint.get("model_name", "bert-base-cased")
    max_length = checkpoint.get("max_length", 128)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    collate_fn = build_parallel_collate_fn(
        tokenizer=tokenizer,
        label2id=tag2idx,
        word2idx=word2idx,
        max_length=max_length,
    )

    model = BertBiLstmCrfNER(
        model_name=model_name,
        num_labels=len(tag2idx),
        word_vocab_size=len(word2idx),
        word_embedding_dim=checkpoint.get("word_embedding_dim", 128),
        lstm_hidden_size=checkpoint.get("lstm_hidden_size", 256),
        dropout=checkpoint.get("dropout", 0.25),
        word_pad_idx=word2idx["<PAD>"],
        id2label=idx2tag,
        label2id=tag2idx,
    ).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    results, prediction_samples = _prepare_prediction_samples(current_batch)
    if not prediction_samples:
        return results

    def collate_fn(batch: list[tuple[str, list[str]]]) -> dict[str, Any]:
        sample_ids, token_sequences = zip(*batch)
        dummy_tags = [["O"] * len(tokens) for tokens in token_sequences]
        model_batch = collate_fn_for_model(list(zip(token_sequences, dummy_tags)))
        model_batch["sample_ids"] = list(sample_ids)
        model_batch["tokens"] = [list(tokens) for tokens in token_sequences]
        return model_batch

    collate_fn_for_model = build_parallel_collate_fn(
        tokenizer=tokenizer,
        label2id=tag2idx,
        word2idx=word2idx,
        max_length=max_length,
    )
    dataloader = _prediction_loader(
        _TokenPredictionDataset(prediction_samples),
        collate_fn,
    )

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            word_input_ids = batch["word_input_ids"].to(device)
            word_attention_mask = batch["word_attention_mask"].to(device)
            first_subword_positions = batch["first_subword_positions"].to(device)
            token_type_ids = batch.get("token_type_ids")
            if token_type_ids is not None:
                token_type_ids = token_type_ids.to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                word_input_ids=word_input_ids,
                word_attention_mask=word_attention_mask,
                first_subword_positions=first_subword_positions,
                token_type_ids=token_type_ids,
            )

            for row_index, (sample_id, tokens) in enumerate(
                zip(batch["sample_ids"], batch["tokens"])
            ):
                pred_ids = outputs["predictions"][row_index]
                kept_token_count = int(word_attention_mask[row_index].sum().item())
                kept_tokens = tokens[:kept_token_count]
                labels = [idx2tag[int(pred_id)] for pred_id in pred_ids]
                results[sample_id] = list(zip(kept_tokens[: len(labels)], labels))

    return results


def model_predict_node(graph_state: GraphState) -> dict:
    """
    节点功能：
        对抽取的未标注样本进行BIO标签预测：
        1. 处理抽取好的样本来满足NER模型的输入
        2. 使用 best_model_path 指向的最好模型来进行 BIO 标签预测
    """
    if not graph_state.current_batch:
        return {"ner_bio_results": {}}

    checkpoint, checkpoint_path = _load_checkpoint(graph_state.best_model_path)
    model_type = _checkpoint_model_type(checkpoint, checkpoint_path)

    if model_type == "bilstm_crf":
        ner_bio_results = _predict_bilstm_crf(checkpoint, graph_state.current_batch)
    elif model_type == "bert_bilstm_crf":
        ner_bio_results = _predict_bert_bilstm_crf(checkpoint, graph_state.current_batch)
    elif model_type == "matscibert_softmax":
        ner_bio_results = _predict_bert_softmax(
            checkpoint,
            graph_state.current_batch,
            use_matscibert=True,
        )
    elif model_type == "bert_softmax":
        ner_bio_results = _predict_bert_softmax(
            checkpoint,
            graph_state.current_batch,
            use_matscibert=False,
        )
    else:
        raise ValueError(f"Unsupported checkpoint model type: {model_type}")

    return {"ner_bio_results": ner_bio_results}


def _append_entity(
    entity_dict: dict[str, list[str]],
    entity_type: str | None,
    entity_tokens: list[str],
) -> None:
    if not entity_type or not entity_tokens:
        return

    entity_text = " ".join(entity_tokens)
    entities = entity_dict.setdefault(entity_type, [])
    if entity_text not in entities:
        entities.append(entity_text)


def _bio_sequence_to_entity_dict(
    bio_sequence: list[tuple[str, str]],
) -> dict[str, list[str]]:
    entity_dict = {}
    current_type = None
    current_tokens = []

    for token, label in bio_sequence:
        if label == "O" or not label:
            _append_entity(entity_dict, current_type, current_tokens)
            current_type = None
            current_tokens = []
            continue

        if "-" not in label:
            _append_entity(entity_dict, current_type, current_tokens)
            current_type = None
            current_tokens = []
            continue

        prefix, entity_type = label.split("-", 1)

        if prefix == "B":
            _append_entity(entity_dict, current_type, current_tokens)
            current_type = entity_type
            current_tokens = [token]
        elif prefix == "I":
            if current_type == entity_type:
                current_tokens.append(token)
            else:
                _append_entity(entity_dict, current_type, current_tokens)
                current_type = entity_type
                current_tokens = [token]
        else:
            _append_entity(entity_dict, current_type, current_tokens)
            current_type = None
            current_tokens = []

    _append_entity(entity_dict, current_type, current_tokens)
    return entity_dict


def BIO2dict_node(graph_state: GraphState) -> dict:
    """
    节点功能：
        根据 ner_bio_results 的结果构造 ner_entity_dicts
    """
    if not graph_state.ner_bio_results:
        return {"ner_entity_dicts": {}}

    return {
        "ner_entity_dicts": {
            sample_id: _bio_sequence_to_entity_dict(bio_sequence)
            for sample_id, bio_sequence in graph_state.ner_bio_results.items()
        }
    }
