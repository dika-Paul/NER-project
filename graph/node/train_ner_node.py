from pathlib import Path
from typing import Any

import torch
import torch.optim as optim
from langgraph.runtime import Runtime
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

from evaluate import (
    evaluate_bert_bilstm_crf,
    evaluate_bert_softmax,
    evaluate_bilstm_crf,
)
from graph.graph_state import GraphContext, GraphState
from models.bert_bilstm_crf import BertBiLstmCrfNER
from models.bert_softmax import BertSoftmaxNER
from models.bilstm_crf import BiLSTM_CRF
from models.matscibert_softmax import MatSciBertSoftmaxNER
from utils.bert.data_utils import (
    NERDataset as BertNERDataset,
    build_collate_fn as build_bert_collate_fn,
    build_tag2idx as build_bert_tag2idx,
    read_conll_2 as read_bert_conll_2,
)
from utils.bert_bilstm_crf.data_utils import (
    NERDataset as BertBiLstmCrfNERDataset,
    build_collate_fn as build_bert_bilstm_crf_collate_fn,
    build_tag2idx as build_bert_bilstm_crf_tag2idx,
    build_vocab as build_bert_bilstm_crf_vocab,
    read_conll_2 as read_bert_bilstm_crf_conll_2,
)
from utils.bilstm_crf.data_utils import (
    NERDataset as BiLSTMCRFDataset,
    build_tag2idx as build_bilstm_crf_tag2idx,
    build_vocab as build_bilstm_crf_vocab,
    collate_fn as bilstm_crf_collate_fn,
    read_conll_2 as read_bilstm_crf_conll_2,
)


GRAPH_DIR = Path(__file__).resolve().parents[1]
SUPPORTED_NER_MODEL_TYPES = {
    "bilstm_crf",
    "bert_bilstm_crf",
    "bert_softmax",
    "matscibert_softmax",
}


def _checkpoint_path(model_type: str, iteration: int) -> Path:
    output_dir = GRAPH_DIR / "model" / model_type
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir / f"{model_type}_iter_{iteration}.pt"


def _metrics(loss: float, precision: float, recall: float, f1: float) -> dict[str, float]:
    return {
        "loss": float(loss),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }


def _ensure_non_empty_dataset(train_sentences: list, valid_sentences: list) -> None:
    if not train_sentences:
        raise ValueError("train_path did not yield any training sentences.")
    if not valid_sentences:
        raise ValueError("valid_path did not yield any validation sentences.")


def _train_bilstm_crf(train_path: str, valid_path: str, iteration: int) -> dict[str, Any]:
    embedding_dim = 300
    hidden_dim = 384
    batch_size = 15
    epochs = 50
    learning_rate = 5e-4
    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_sentences, train_tags = read_bilstm_crf_conll_2(train_path)
    valid_sentences, valid_tags = read_bilstm_crf_conll_2(valid_path)
    _ensure_non_empty_dataset(train_sentences, valid_sentences)

    word2idx = build_bilstm_crf_vocab(train_sentences)
    tag2idx, idx2tag = build_bilstm_crf_tag2idx(train_tags)

    train_data = BiLSTMCRFDataset(train_sentences, train_tags, word2idx, tag2idx)
    valid_data = BiLSTMCRFDataset(valid_sentences, valid_tags, word2idx, tag2idx)

    train_loader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=bilstm_crf_collate_fn,
    )
    valid_loader = DataLoader(
        valid_data,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=bilstm_crf_collate_fn,
    )

    model = BiLSTM_CRF(
        vocab_size=len(word2idx),
        tag_to_ix=tag2idx,
        embedding_dim=embedding_dim,
        hidden_dim=hidden_dim,
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=3,
    )

    model_path = _checkpoint_path("bilstm_crf", iteration)
    best_valid_f1 = -1.0
    best_metrics = _metrics(float("inf"), -1.0, -1.0, -1.0)

    for epoch in range(epochs):
        model.train()
        total_train_loss = 0.0
        total_train_sentences = 0

        for sentences, tags, lengths in train_loader:
            sentences = sentences.to(device)
            tags = tags.to(device)
            lengths = lengths.to(device)

            optimizer.zero_grad()
            loss = model.neg_log_likelihood(sentences, tags, lengths)
            batch_sentence_count = sentences.size(0)

            loss.backward()
            optimizer.step()

            total_train_loss += loss.item() * batch_sentence_count
            total_train_sentences += batch_sentence_count

        train_loss = total_train_loss / total_train_sentences
        valid_loss, valid_precision, valid_recall, valid_f1, _, _ = evaluate_bilstm_crf(
            model,
            valid_loader,
            idx2tag,
            device,
        )

        print(f"\nEpoch {epoch + 1}/{epochs}")
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Valid Loss: {valid_loss:.4f}")
        print(f"Valid Precision: {valid_precision:.4f}")
        print(f"Valid Recall: {valid_recall:.4f}")
        print(f"Valid F1: {valid_f1:.4f}")

        if valid_f1 > best_valid_f1:
            best_valid_f1 = valid_f1
            best_metrics = _metrics(valid_loss, valid_precision, valid_recall, valid_f1)
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "word2idx": word2idx,
                    "tag2idx": tag2idx,
                    "idx2tag": idx2tag,
                    "embedding_dim": embedding_dim,
                    "hidden_dim": hidden_dim,
                    "metrics": best_metrics,
                    "iteration": iteration,
                },
                model_path,
            )
            print(f"保存当前最优模型: {model_path}")

        scheduler.step(valid_f1)

    return {
        "model_path": str(model_path),
        "metrics": best_metrics,
    }


def _train_bert_softmax(
    train_path: str,
    valid_path: str,
    iteration: int,
    *,
    model_type: str = "bert_softmax",
    model_name: str = "bert-base-cased",
    model_cls: type[BertSoftmaxNER] = BertSoftmaxNER,
) -> dict[str, Any]:

    max_length = 128
    batch_size = 16
    epochs = 5
    learning_rate = 3e-5
    weight_decay = 0.01
    warmup_ratio = 0.1
    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_sentences, train_tags = read_bert_conll_2(train_path)
    valid_sentences, valid_tags = read_bert_conll_2(valid_path)
    _ensure_non_empty_dataset(train_sentences, valid_sentences)

    tag2idx, idx2tag = build_bert_tag2idx(train_tags)

    train_dataset = BertNERDataset(train_sentences, train_tags)
    valid_dataset = BertNERDataset(valid_sentences, valid_tags)

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    collate_fn = build_bert_collate_fn(
        tokenizer=tokenizer,
        label2id=tag2idx,
        max_length=max_length,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
    )

    model = model_cls(
        model_name=model_name,
        num_labels=len(tag2idx),
        id2label=idx2tag,
        label2id=tag2idx,
    ).to(device)

    optimizer = AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    total_steps = len(train_loader) * epochs
    warmup_steps = int(warmup_ratio * total_steps)
    scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    model_path = _checkpoint_path(model_type, iteration)
    best_valid_f1 = -1.0
    best_metrics = _metrics(float("inf"), -1.0, -1.0, -1.0)

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0

        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            token_type_ids = batch.get("token_type_ids")

            if token_type_ids is not None:
                token_type_ids = token_type_ids.to(device)

            optimizer.zero_grad()
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                token_type_ids=token_type_ids,
            )

            loss = outputs["loss"]
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            scheduler.step()

            total_loss += loss.item()

        train_loss = total_loss / len(train_loader)
        valid_loss, valid_precision, valid_recall, valid_f1, _, _ = evaluate_bert_softmax(
            model=model,
            dataloader=valid_loader,
            id2label=idx2tag,
            device=device,
        )

        print(f"\nEpoch {epoch + 1}/{epochs}")
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Valid Loss: {valid_loss:.4f}")
        print(f"Valid Precision: {valid_precision:.4f}")
        print(f"Valid Recall: {valid_recall:.4f}")
        print(f"Valid F1: {valid_f1:.4f}")

        if valid_f1 > best_valid_f1:
            best_valid_f1 = valid_f1
            best_metrics = _metrics(valid_loss, valid_precision, valid_recall, valid_f1)
            torch.save(
                {
                    "model": model.state_dict(),
                    "tag2idx": tag2idx,
                    "idx2tag": idx2tag,
                    "model_name": model_name,
                    "max_length": max_length,
                    "metrics": best_metrics,
                    "iteration": iteration,
                },
                model_path,
            )
            print(f"保存当前最优模型: {model_path}")

    return {
        "model_path": str(model_path),
        "metrics": best_metrics,
    }


def _set_bert_trainable(model: BertBiLstmCrfNER, trainable: bool) -> None:
    for param in model.bert.parameters():
        param.requires_grad = trainable


def _train_bert_bilstm_crf(
    train_path: str,
    valid_path: str,
    iteration: int,
) -> dict[str, Any]:
    model_name = "bert-base-cased"
    max_length = 128
    batch_size = 8
    joint_train_epochs = 5
    bilstm_only_epochs = 30
    word_embedding_dim = 128
    lstm_hidden_size = 256
    dropout = 0.25
    bert_learning_rate = 3e-5
    other_learning_rate = 1e-3
    weight_decay = 0.01
    warmup_ratio = 0.1
    total_epochs = joint_train_epochs + bilstm_only_epochs
    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_sentences, train_tags = read_bert_bilstm_crf_conll_2(train_path)
    valid_sentences, valid_tags = read_bert_bilstm_crf_conll_2(valid_path)
    _ensure_non_empty_dataset(train_sentences, valid_sentences)

    word2idx = build_bert_bilstm_crf_vocab(train_sentences)
    tag2idx, idx2tag = build_bert_bilstm_crf_tag2idx(train_tags)

    train_dataset = BertBiLstmCrfNERDataset(train_sentences, train_tags)
    valid_dataset = BertBiLstmCrfNERDataset(valid_sentences, valid_tags)

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    collate_fn = build_bert_bilstm_crf_collate_fn(
        tokenizer=tokenizer,
        label2id=tag2idx,
        word2idx=word2idx,
        max_length=max_length,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
    )

    model = BertBiLstmCrfNER(
        model_name=model_name,
        num_labels=len(tag2idx),
        word_vocab_size=len(word2idx),
        word_embedding_dim=word_embedding_dim,
        lstm_hidden_size=lstm_hidden_size,
        dropout=dropout,
        word_pad_idx=word2idx["<PAD>"],
        id2label=idx2tag,
        label2id=tag2idx,
    ).to(device)

    def build_optimizer_and_scheduler(
        train_bert: bool,
        num_epochs: int,
    ) -> tuple[AdamW, Any]:
        param_groups = []

        if train_bert:
            param_groups.append(
                {"params": model.bert.parameters(), "lr": bert_learning_rate}
            )

        param_groups.extend(
            [
                {"params": model.word_embeddings.parameters(), "lr": other_learning_rate},
                {"params": model.bilstm.parameters(), "lr": other_learning_rate},
                {"params": model.classifier.parameters(), "lr": other_learning_rate},
                {"params": model.crf.parameters(), "lr": other_learning_rate},
            ]
        )

        optimizer = AdamW(param_groups, weight_decay=weight_decay)
        total_steps = len(train_loader) * num_epochs
        warmup_steps = int(warmup_ratio * total_steps)
        scheduler = get_linear_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
        )
        return optimizer, scheduler

    _set_bert_trainable(model, True)
    optimizer, scheduler = build_optimizer_and_scheduler(
        train_bert=True,
        num_epochs=joint_train_epochs,
    )
    bert_is_trainable = True

    model_path = _checkpoint_path("bert_bilstm_crf", iteration)
    best_valid_f1 = -1.0
    best_metrics = _metrics(float("inf"), -1.0, -1.0, -1.0)

    for epoch in range(total_epochs):
        if epoch == joint_train_epochs and bert_is_trainable:
            _set_bert_trainable(model, False)
            optimizer, scheduler = build_optimizer_and_scheduler(
                train_bert=False,
                num_epochs=bilstm_only_epochs,
            )
            bert_is_trainable = False
            print("\n切换到第二阶段：冻结 BERT，仅继续训练 BiLSTM、分类层和 CRF。")

        model.train()
        if not bert_is_trainable:
            model.bert.eval()

        total_loss = 0.0

        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            word_input_ids = batch["word_input_ids"].to(device)
            word_attention_mask = batch["word_attention_mask"].to(device)
            first_subword_positions = batch["first_subword_positions"].to(device)
            labels = batch["labels"].to(device)
            token_type_ids = batch.get("token_type_ids")

            if token_type_ids is not None:
                token_type_ids = token_type_ids.to(device)

            optimizer.zero_grad()
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                word_input_ids=word_input_ids,
                word_attention_mask=word_attention_mask,
                first_subword_positions=first_subword_positions,
                labels=labels,
                token_type_ids=token_type_ids,
            )

            loss = outputs["loss"]
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            scheduler.step()

            total_loss += loss.item()

        train_loss = total_loss / len(train_loader)
        valid_loss, valid_precision, valid_recall, valid_f1, _, _ = (
            evaluate_bert_bilstm_crf(
                model=model,
                dataloader=valid_loader,
                id2label=idx2tag,
                device=device,
            )
        )

        stage_name = "阶段1（联合训练）" if bert_is_trainable else "阶段2（冻结 BERT）"
        print(f"\nEpoch {epoch + 1}/{total_epochs} - {stage_name}")
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Valid Loss: {valid_loss:.4f}")
        print(f"Valid Precision: {valid_precision:.4f}")
        print(f"Valid Recall: {valid_recall:.4f}")
        print(f"Valid F1: {valid_f1:.4f}")

        if valid_f1 > best_valid_f1:
            best_valid_f1 = valid_f1
            best_metrics = _metrics(valid_loss, valid_precision, valid_recall, valid_f1)
            torch.save(
                {
                    "model": model.state_dict(),
                    "word2idx": word2idx,
                    "tag2idx": tag2idx,
                    "idx2tag": idx2tag,
                    "model_name": model_name,
                    "max_length": max_length,
                    "word_embedding_dim": word_embedding_dim,
                    "lstm_hidden_size": lstm_hidden_size,
                    "dropout": dropout,
                    "joint_train_epochs": joint_train_epochs,
                    "bilstm_only_epochs": bilstm_only_epochs,
                    "bert_learning_rate": bert_learning_rate,
                    "other_learning_rate": other_learning_rate,
                    "best_stage": stage_name,
                    "metrics": best_metrics,
                    "iteration": iteration,
                },
                model_path,
            )
            print(f"保存当前最优模型: {model_path}")

    return {
        "model_path": str(model_path),
        "metrics": best_metrics,
    }


def _get_context_ner_model(runtime: Runtime[GraphContext]) -> str:
    context = getattr(runtime, "context", None)
    if context is None:
        raise ValueError("runtime.context must provide ner_model as a string.")

    if isinstance(context, dict):
        ner_model = context.get("ner_model")
    else:
        ner_model = getattr(context, "ner_model", None)

    if ner_model is None:
        raise ValueError("runtime.context must provide ner_model as a string.")

    if not isinstance(ner_model, str):
        raise ValueError(
            "runtime.context ner_model must be a string, "
            f"got {type(ner_model).__name__}."
        )

    ner_model = ner_model.strip().lower()
    if not ner_model:
        raise ValueError("runtime.context ner_model cannot be empty.")

    if ner_model not in SUPPORTED_NER_MODEL_TYPES:
        supported_types = ", ".join(sorted(SUPPORTED_NER_MODEL_TYPES))
        raise ValueError(
            f"Unsupported ner_model type: {ner_model}. "
            f"Supported values are: {supported_types}."
        )

    return ner_model


def train_ner_node(
    graph_state: GraphState,
    runtime: Runtime[GraphContext],
) -> dict:
    """
    节点功能：
        完成数据集的加载，
        根据 GraphContext 中的 ner_model，
        选择合适的训练函数对模型进行训练。
    """
    print(f"开始第{graph_state.iteration+1}轮训练：\n\n\n")

    ner_model = _get_context_ner_model(runtime)

    if ner_model == "bilstm_crf":
        result = _train_bilstm_crf(
            graph_state.train_path,
            graph_state.valid_path,
            graph_state.iteration,
        )
    elif ner_model == "bert_bilstm_crf":
        result = _train_bert_bilstm_crf(
            graph_state.train_path,
            graph_state.valid_path,
            graph_state.iteration,
        )
    elif ner_model == "matscibert_softmax":
        result = _train_bert_softmax(
            graph_state.train_path,
            graph_state.valid_path,
            graph_state.iteration,
            model_type="matscibert_softmax",
            model_name="m3rg-iitd/matscibert",
            model_cls=MatSciBertSoftmaxNER,
        )
    elif ner_model == "bert_softmax":
        result = _train_bert_softmax(
            graph_state.train_path,
            graph_state.valid_path,
            graph_state.iteration,
            model_type="bert_softmax",
            model_name="bert-base-cased",
            model_cls=BertSoftmaxNER,
        )
    else:
        supported_types = ", ".join(sorted(SUPPORTED_NER_MODEL_TYPES))
        raise ValueError(
            f"Unsupported ner_model type: {ner_model}. "
            f"Supported values are: {supported_types}."
        )

    return {
        "ner_model_path": result["model_path"],
        "current_metrics": result["metrics"],
        "previous_metrics": result["metrics"]
    }
