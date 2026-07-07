"""Data pipeline for BERT pretraining (Wikipedia + BookCorpusOpen, streaming)."""

import itertools
import queue
import threading

import jax.numpy as jnp
import numpy as np


def get_tokenizer(vocab_size=30522):
    """Load bert-base-uncased tokenizer from HuggingFace."""
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained("bert-base-uncased")


def mlm_collate(batch_ids, tokenizer, mlm_prob=0.15, rng=None):
    """Apply BERT-style MLM masking to a batch of token IDs.

    80% [MASK], 10% random token, 10% unchanged.

    Args:
        batch_ids: np.ndarray of shape (batch_size, seq_len)
        tokenizer: HuggingFace tokenizer (needs mask_token_id, vocab_size)
        mlm_prob: probability of masking each token
        rng: numpy random state (default: np.random.default_rng())

    Returns:
        input_ids: np.ndarray (batch_size, seq_len) — masked inputs
        labels: np.ndarray (batch_size, seq_len) — -100 for non-masked positions
    """
    if rng is None:
        rng = np.random.default_rng()

    input_ids = batch_ids.copy()
    labels = np.full_like(batch_ids, -100)

    # Determine which tokens to mask
    prob_matrix = rng.random(batch_ids.shape)
    # Don't mask special tokens (CLS=101, SEP=102, PAD=0)
    special_mask = (batch_ids == tokenizer.cls_token_id) | \
                   (batch_ids == tokenizer.sep_token_id) | \
                   (batch_ids == tokenizer.pad_token_id)
    prob_matrix[special_mask] = 1.0  # won't be selected

    masked_indices = prob_matrix < mlm_prob
    labels[~masked_indices] = -100
    labels[masked_indices] = batch_ids[masked_indices]

    # 80% of masked: replace with [MASK]
    replace_mask_indices = masked_indices & (rng.random(batch_ids.shape) < 0.8)
    input_ids[replace_mask_indices] = tokenizer.mask_token_id

    # 10% of masked: replace with random token
    random_indices = masked_indices & ~replace_mask_indices & (rng.random(batch_ids.shape) < 0.5)
    random_tokens = rng.integers(0, tokenizer.vocab_size, size=batch_ids.shape)
    input_ids[random_indices] = random_tokens[random_indices]

    # Remaining 10%: keep original (already in input_ids)

    return input_ids, labels


def create_dataset(tokenizer, seq_len=128, seed=42):
    """Load Wikipedia + BookCorpusOpen via HuggingFace datasets (streaming).

    Tokenizes, concatenates, and chunks into seq_len windows.
    Returns an iterable yielding np.ndarray of shape (seq_len,).
    """
    from datasets import load_dataset, interleave_datasets

    wiki = load_dataset("wikimedia/wikipedia", "20231101.en", split="train", streaming=True)
    books = load_dataset("lucadiliello/bookcorpusopen", split="train", streaming=True)

    # Extract text field (both have "text")
    wiki = wiki.remove_columns([c for c in wiki.column_names if c != "text"])
    books = books.remove_columns([c for c in books.column_names if c != "text"])

    combined = interleave_datasets([wiki, books], seed=seed)
    combined = combined.shuffle(seed=seed, buffer_size=10000)

    def tokenize_and_chunk():
        buffer = []
        for example in combined:
            encoded = tokenizer(
                example["text"],
                add_special_tokens=False,
                truncation=False,
                return_attention_mask=False,
            )["input_ids"]
            buffer.extend(encoded)

            while len(buffer) >= seq_len - 2:
                chunk = [tokenizer.cls_token_id] + buffer[:seq_len - 2] + [tokenizer.sep_token_id]
                yield np.array(chunk, dtype=np.int32)
                buffer = buffer[seq_len - 2:]

    return tokenize_and_chunk()


def create_dataloader(dataset, batch_size, tokenizer, mlm_prob=0.15, seed=42,
                      prefetch=8):
    """Iterate over dataset, yield JAX arrays of (input_ids, labels).

    Uses a background thread to prefetch batches so data loading overlaps
    with TPU computation.

    Args:
        dataset: iterable of np.ndarray chunks (seq_len,)
        batch_size: number of examples per batch
        tokenizer: HuggingFace tokenizer for MLM masking
        mlm_prob: MLM masking probability
        seed: random seed for masking
        prefetch: number of batches to prefetch in background

    Yields:
        (input_ids, labels) as jnp.ndarray of shape (batch_size, seq_len)
    """
    q = queue.Queue(maxsize=prefetch)
    sentinel = None

    def producer():
        rng = np.random.default_rng(seed)
        while True:
            batch = list(itertools.islice(dataset, batch_size))
            if len(batch) < batch_size:
                break
            batch_ids = np.stack(batch)
            input_ids, labels = mlm_collate(batch_ids, tokenizer,
                                            mlm_prob=mlm_prob, rng=rng)
            q.put((jnp.array(input_ids), jnp.array(labels)))
        q.put(sentinel)

    thread = threading.Thread(target=producer, daemon=True)
    thread.start()

    while True:
        item = q.get()
        if item is None:
            break
        yield item


# ---------------------------------------------------------------------------
# GLUE task loading (SST-2, MNLI)
# ---------------------------------------------------------------------------

GLUE_TASKS = {
    "sst2": {"num_classes": 2, "sentence_keys": ("sentence",)},
    "mnli": {"num_classes": 3, "sentence_keys": ("premise", "hypothesis")},
}


def load_glue_task(task_name, tokenizer, seq_len=128, split="train"):
    """Load a GLUE task and return tokenized examples.

    Args:
        task_name: "sst2" or "mnli"
        tokenizer: HuggingFace tokenizer
        seq_len: max sequence length (pad/truncate)
        split: dataset split ("train", "validation", "validation_matched",
               "validation_mismatched")

    Returns:
        input_ids: np.ndarray (num_examples, seq_len)
        labels: np.ndarray (num_examples,)
    """
    from datasets import load_dataset

    task_info = GLUE_TASKS[task_name]
    ds = load_dataset("glue", task_name, split=split)

    all_input_ids = []
    all_labels = []

    keys = task_info["sentence_keys"]
    for example in ds:
        if len(keys) == 1:
            encoded = tokenizer(
                example[keys[0]],
                max_length=seq_len,
                padding="max_length",
                truncation=True,
                return_attention_mask=False,
            )
        else:
            encoded = tokenizer(
                example[keys[0]],
                example[keys[1]],
                max_length=seq_len,
                padding="max_length",
                truncation=True,
                return_attention_mask=False,
            )

        all_input_ids.append(encoded["input_ids"])
        all_labels.append(example["label"])

    return np.array(all_input_ids, dtype=np.int32), np.array(all_labels, dtype=np.int32)


def create_glue_dataloader(input_ids, labels, batch_size, shuffle=True, seed=42):
    """Yield batches of (input_ids, labels) as JAX arrays.

    Args:
        input_ids: np.ndarray (num_examples, seq_len)
        labels: np.ndarray (num_examples,)
        batch_size: examples per batch
        shuffle: whether to shuffle each epoch
        seed: random seed for shuffling

    Yields:
        (input_ids_batch, labels_batch) as jnp.ndarray
    """
    n = len(labels)
    rng = np.random.default_rng(seed)

    indices = np.arange(n)
    if shuffle:
        rng.shuffle(indices)

    for start in range(0, n - batch_size + 1, batch_size):
        idx = indices[start:start + batch_size]
        yield jnp.array(input_ids[idx]), jnp.array(labels[idx])
