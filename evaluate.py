import torch
from seqeval.metrics import precision_score, recall_score, f1_score

@torch.no_grad()
def evaluate_bilstm_crf(model, data_loader, idx2tag, device):
    """
    在验证集或测试集上评估模型

    说明：
        以整个 batch 为单位计算 loss，

        具体做法：
            1. 调用 model.neg_log_likelihood(sentences, tags, lengths)
               直接得到当前 batch 的平均 loss
            2. 调用 model(sentences, lengths)
               直接得到当前 batch 中每个句子的预测标签序列
    """
    model.eval()

    total_loss = 0.0
    total_sentences = 0

    all_true_labels = []
    all_pred_labels = []

    with torch.no_grad():
        for sentences, tags, lengths in data_loader:
            sentences = sentences.to(device)
            tags = tags.to(device)
            lengths = lengths.to(device)

            batch_size = sentences.size(0)

            # 直接以整个 batch 为单位计算平均 loss
            loss = model.neg_log_likelihood(sentences, tags, lengths)
            total_loss += loss.item() * batch_size
            total_sentences += batch_size

            # 直接做 batch 解码
            batch_pred_ids = model(sentences, lengths)

            # 逐句整理真实标签和预测标签，用于计算 precision / recall / f1
            for i in range(batch_size):
                seq_len = lengths[i].item()

                # 只取真实有效长度范围内的标签
                tags_i = tags[i, :seq_len]
                pred_ids = batch_pred_ids[i]

                # 转成标签文本
                true_labels = [idx2tag[int(x)] for x in tags_i.cpu().tolist()]
                pred_labels = [idx2tag[int(x)] for x in pred_ids]

                all_true_labels.append(true_labels)
                all_pred_labels.append(pred_labels)

    avg_loss = total_loss / total_sentences if total_sentences > 0 else 0.0
    precision = precision_score(all_true_labels, all_pred_labels)
    recall = recall_score(all_true_labels, all_pred_labels)
    f1 = f1_score(all_true_labels, all_pred_labels)

    return avg_loss, precision, recall, f1, all_true_labels, all_pred_labels


@torch.no_grad()
def evaluate_bert_softmax(model, dataloader, id2label, device):
    """
    在验证集或测试集上评估 BERT+Softmax 模型

    说明：
        1. 以整个 batch 为单位计算平均 loss
        2. 以整个 batch 为单位得到预测结果
        3. 只统计 labels != -100 的位置
           （即真实词的首个 subword）
    """
    model.eval()

    total_loss = 0.0
    total_sentences = 0

    all_true_labels = []
    all_pred_labels = []

    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        token_type_ids = batch.get("token_type_ids")

        if token_type_ids is not None:
            token_type_ids = token_type_ids.to(device)

        batch_size = input_ids.size(0)

        # 整个 batch 一次性前向传播，直接得到 loss 和 logits
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            token_type_ids=token_type_ids
        )

        loss = outputs["loss"]
        logits = outputs["logits"]

        total_loss += loss.item() * batch_size
        total_sentences += batch_size

        # 取每个位置概率最大的标签 id
        batch_pred_ids = torch.argmax(logits, dim=-1)

        # 逐句整理真实标签和预测标签
        for i in range(batch_size):
            label_seq = labels[i]
            pred_seq = batch_pred_ids[i]

            true_labels = []
            pred_labels = []

            for gold_id, pred_id in zip(label_seq, pred_seq):
                gold_id = int(gold_id.item())
                pred_id = int(pred_id.item())

                # -100 的位置不参与评估
                if gold_id == -100:
                    continue

                true_labels.append(id2label[gold_id])
                pred_labels.append(id2label[pred_id])

            all_true_labels.append(true_labels)
            all_pred_labels.append(pred_labels)

    avg_loss = total_loss / total_sentences if total_sentences > 0 else 0.0
    precision = precision_score(all_true_labels, all_pred_labels)
    recall = recall_score(all_true_labels, all_pred_labels)
    f1 = f1_score(all_true_labels, all_pred_labels)

    return avg_loss, precision, recall, f1, all_true_labels, all_pred_labels


@torch.no_grad()
def evaluate_bert_bilstm_crf(model, dataloader, id2label, device):
    """
    在验证集或测试集上评估并行 BERT + BiLSTM + CRF 模型。

    说明：
        1. loss 按 batch 维度求平均，与当前项目中其它评估函数保持一致。
        2. 指标在 CRF 解码后的词级标签序列上计算。
    """
    model.eval()

    total_loss = 0.0
    total_sentences = 0

    all_true_labels = []
    all_pred_labels = []

    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        word_attention_mask = batch["word_attention_mask"].to(device)
        first_subword_positions = batch["first_subword_positions"].to(device)
        labels = batch["labels"].to(device)
        token_type_ids = batch.get("token_type_ids")

        if token_type_ids is not None:
            token_type_ids = token_type_ids.to(device)

        batch_size = input_ids.size(0)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            word_attention_mask=word_attention_mask,
            first_subword_positions=first_subword_positions,
            labels=labels,
            token_type_ids=token_type_ids,
        )

        loss = outputs["loss"]
        batch_pred_ids = outputs["predictions"]

        total_loss += loss.item() * batch_size
        total_sentences += batch_size

        for i in range(batch_size):
            seq_len = int(word_attention_mask[i].sum().item())

            gold_ids = labels[i, :seq_len].tolist()
            pred_ids = batch_pred_ids[i]

            true_labels = [id2label[int(gold_id)] for gold_id in gold_ids]
            pred_labels = [id2label[int(pred_id)] for pred_id in pred_ids]

            all_true_labels.append(true_labels)
            all_pred_labels.append(pred_labels)

    avg_loss = total_loss / total_sentences if total_sentences > 0 else 0.0
    precision = precision_score(all_true_labels, all_pred_labels)
    recall = recall_score(all_true_labels, all_pred_labels)
    f1 = f1_score(all_true_labels, all_pred_labels)

    return avg_loss, precision, recall, f1, all_true_labels, all_pred_labels
