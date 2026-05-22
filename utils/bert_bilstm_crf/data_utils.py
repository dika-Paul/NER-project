import torch
from torch.utils.data import Dataset


def read_conll_4(path):
    """
    Read a CoNLL-style file with at least four columns.
    The first column is treated as the token and the last column as the label.
    """
    sentences = []
    tags = []

    words = []
    labels = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if line == "":
                if words:
                    sentences.append(words)
                    tags.append(labels)
                    words, labels = [], []
                continue

            if line.startswith("-DOCSTART-"):
                continue

            parts = line.split()
            if len(parts) < 4:
                continue

            words.append(parts[0])
            labels.append(parts[-1])

    if words:
        sentences.append(words)
        tags.append(labels)

    return sentences, tags


def read_conll_2(path):
    """Read a two-column CoNLL-style file: token and label."""
    sentences = []
    tags = []

    words = []
    labels = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if not line:
                if words:
                    sentences.append(words)
                    tags.append(labels)
                    words, labels = [], []
                continue

            parts = line.split()
            if len(parts) != 2:
                continue

            word, label = parts
            words.append(word)
            labels.append(label)

    if words:
        sentences.append(words)
        tags.append(labels)

    return sentences, tags


def build_vocab(sentences, min_freq=1):
    """
    Legacy helper kept for older notebooks/checkpoints. The sequential
    BERT-BiLSTM-CRF model no longer uses a word-level vocabulary.
    """
    word_count = {}
    for sentence in sentences:
        for word in sentence:
            word_count[word] = word_count.get(word, 0) + 1

    word2idx = {
        "<PAD>": 0,
        "<UNK>": 1,
    }

    for word, count in word_count.items():
        if count >= min_freq:
            word2idx[word] = len(word2idx)

    return word2idx


def build_tag2idx(tags_list):
    """Build label-to-id and id-to-label mappings."""
    tag2idx = {}

    for tags in tags_list:
        for tag in tags:
            if tag not in tag2idx:
                tag2idx[tag] = len(tag2idx)

    idx2tag = {idx: tag for tag, idx in tag2idx.items()}
    return tag2idx, idx2tag


def encode_sentence(sentence, word2idx):
    """
    Legacy helper kept for older code paths. The sequential model does not call it.
    """
    unk_id = word2idx["<UNK>"]
    return [word2idx.get(word, unk_id) for word in sentence]


class NERDataset(Dataset):
    """
    Store raw token and label sequences. Tokenization and label alignment happen
    inside collate_fn so that subword truncation stays synchronized with labels.
    """

    def __init__(self, sentences, tags):
        self.sentences = sentences
        self.tags = tags

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        return self.sentences[idx], self.tags[idx]


def build_collate_fn(tokenizer, label2id, max_length=128):
    """
    Build a collate_fn for Tokenizer -> BERT -> BiLSTM -> CRF.

    Returned tensors:
        input_ids / attention_mask / token_type_ids:
            Subword-level BERT inputs from the tokenizer.
        first_subword_positions:
            Word-level positions pointing to the first subword of each token.
        word_attention_mask:
            Word-level mask used by BiLSTM packing and CRF decoding.
        labels:
            Word-level labels aligned with first_subword_positions.
    """

    def collate_fn(batch):
        batch_sentences, batch_tags = zip(*batch)

        encodings = tokenizer(
            list(batch_sentences),
            is_split_into_words=True,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )

        encoded_examples = []
        max_words = 0

        for i, (sentence, tags) in enumerate(zip(batch_sentences, batch_tags)):
            word_ids = encodings.word_ids(batch_index=i)

            first_subword_positions = []
            previous_word_id = None

            for token_position, word_id in enumerate(word_ids):
                if word_id is None:
                    continue

                if word_id != previous_word_id:
                    first_subword_positions.append(token_position)

                previous_word_id = word_id

            valid_word_count = len(first_subword_positions)

            if valid_word_count == 0:
                raise ValueError(
                    "No tokens remain after max_length truncation. "
                    "Increase max_length for the BERT-BiLSTM-CRF model."
                )

            kept_tags = list(tags[:valid_word_count])

            encoded_examples.append(
                {
                    "labels": [label2id[tag] for tag in kept_tags],
                    "first_subword_positions": first_subword_positions,
                    "word_count": valid_word_count,
                }
            )
            max_words = max(max_words, valid_word_count)

        batch_size = len(encoded_examples)

        labels = torch.zeros((batch_size, max_words), dtype=torch.long)
        first_subword_positions = torch.zeros((batch_size, max_words), dtype=torch.long)
        word_attention_mask = torch.zeros((batch_size, max_words), dtype=torch.long)

        for i, example in enumerate(encoded_examples):
            word_count = example["word_count"]

            labels[i, :word_count] = torch.tensor(example["labels"], dtype=torch.long)
            first_subword_positions[i, :word_count] = torch.tensor(
                example["first_subword_positions"], dtype=torch.long
            )
            word_attention_mask[i, :word_count] = 1

        batch_dict = {
            "input_ids": encodings["input_ids"],
            "attention_mask": encodings["attention_mask"],
            "word_attention_mask": word_attention_mask,
            "first_subword_positions": first_subword_positions,
            "labels": labels,
        }

        if "token_type_ids" in encodings:
            batch_dict["token_type_ids"] = encodings["token_type_ids"]

        return batch_dict

    return collate_fn
