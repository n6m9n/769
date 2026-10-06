"""Frequency-based baselines (Hugging Face `tokenizers`): byte-level BPE and Unigram.

Same pretokenizer regex as the split-tree tokenizer, so differences come from
vocabulary selection, not pretokenization. Multilingual mixing uses the standard
temperature-sampling knob: q_l proportional to p_l^alpha (alpha=1 natural, alpha->0 uniform).
"""
from __future__ import annotations

import random
from pathlib import Path
from typing import Iterator

from tokenizers import Regex, Tokenizer, decoders, models, pre_tokenizers, trainers

from motok.pretokenize import GPT4O_REGEX_16, iter_lines


def _pretok():
    return pre_tokenizers.Sequence([
        pre_tokenizers.Split(Regex(GPT4O_REGEX_16), behavior="isolated"),
        pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False),
    ])


def temperature_mixture(files: dict[str, str], alpha: float, total_lines: int | None = None,
                        seed: int = 0) -> Iterator[str]:
    """Yield lines sampled so that language l has share proportional to (bytes_l)^alpha."""
    rng = random.Random(seed)
    data = {l: list(iter_lines(p)) for l, p in files.items()}
    sizes = {l: sum(len(x.encode()) for x in lines) for l, lines in data.items()}
    w = {l: sizes[l] ** alpha for l in data}
    Z = sum(w.values())
    total_bytes = sum(sizes.values()) if total_lines is None else None
    out = []
    for l, lines in data.items():
        if not lines:
            continue
        target_bytes = (w[l] / Z) * (total_bytes if total_bytes else sizes[l])
        avg = sizes[l] / len(lines)
        n = int(target_bytes / avg) if total_bytes else int(total_lines * w[l] / Z)
        reps, rem = divmod(n, len(lines))
        out.extend(lines * reps + rng.sample(lines, rem))
    rng.shuffle(out)
    yield from out


def train_hf(kind: str, lines: Iterator[str], vocab_size: int, save_path: str | None = None) -> Tokenizer:
    if kind == "bpe":
        tok = Tokenizer(models.BPE())
        trainer = trainers.BpeTrainer(vocab_size=vocab_size, initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
                                      show_progress=False)
    elif kind == "unigram":
        tok = Tokenizer(models.Unigram())
        trainer = trainers.UnigramTrainer(vocab_size=vocab_size, initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
                                          show_progress=False, unk_token=None, max_piece_length=32)
    else:
        raise ValueError(kind)
    tok.pre_tokenizer = _pretok()
    tok.decoder = decoders.ByteLevel()
    tok.train_from_iterator(lines, trainer=trainer)
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        tok.save(save_path)
    return tok


class HFWrapper:
    """Uniform interface used by the evaluation code."""

    def __init__(self, tok: Tokenizer, name: str = "hf"):
        self.tok, self.name = tok, name

    @classmethod
    def load(cls, path, name=None):
        return cls(Tokenizer.from_file(str(path)), name or Path(path).stem)

    @property
    def vocab_size(self) -> int:
        return self.tok.get_vocab_size()

    def encode(self, text: str) -> list[str]:
        return self.tok.encode(text, add_special_tokens=False).tokens

    def encode_with_offsets(self, text: str):
        e = self.tok.encode(text, add_special_tokens=False)
        return e.tokens, e.offsets

    def decode_tokens(self, text: str) -> str:
        return self.tok.decode(self.tok.encode(text, add_special_tokens=False).ids)


class SplitTreeWrapper:
    def __init__(self, st, name="toast"):
        self.st, self.name = st, name

    @property
    def vocab_size(self) -> int:
        return len(self.st.vocab) + 256

    def encode(self, text):
        return self.st.encode(text)

    def encode_with_offsets(self, text):
        return self.st.encode_with_offsets(text)

    def decode_tokens(self, text):
        return self.st.decode(self.st.encode(text))
