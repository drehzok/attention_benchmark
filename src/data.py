"""Data pipeline for BERT pretraining (Wikipedia + BookCorpusOpen, streaming)."""

import itertools

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
    books = load_dataset("bookcorpusopen", split="train", streaming=True, trust_remote_code=True)

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

            while len(buffer) >= seq_len:
                yield np.array(buffer[:seq_len], dtype=np.int32)
                buffer = buffer[seq_len:]

    return tokenize_and_chunk()


def create_dataloader(dataset, batch_size, tokenizer, mlm_prob=0.15, seed=42):
    """Iterate over dataset, yield JAX arrays of (input_ids, labels).

    Args:
        dataset: iterable of np.ndarray chunks (seq_len,)
        batch_size: number of examples per batch
        tokenizer: HuggingFace tokenizer for MLM masking
        mlm_prob: MLM masking probability
        seed: random seed for masking

    Yields:
        (input_ids, labels) as jnp.ndarray of shape (batch_size, seq_len)
    """
    rng = np.random.default_rng(seed)

    while True:
        batch = list(itertools.islice(dataset, batch_size))
        if len(batch) < batch_size:
            break

        batch_ids = np.stack(batch)
        input_ids, labels = mlm_collate(batch_ids, tokenizer, mlm_prob=mlm_prob, rng=rng)

        yield jnp.array(input_ids), jnp.array(labels)
