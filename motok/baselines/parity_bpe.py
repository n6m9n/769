"""Parity-aware BPE baseline (Foroutan et al., ACL 2026; github.com/swiss-ai/parity-aware-bpe).

We run the authors' learner (`ext/parity-aware-bpe`) unchanged, except for its inputs:
  * pretokenization: our GPT-4o regex + byte-level mapping (same as every other tokenizer here),
    passed as word-count dictionaries (`--dict-input` format), so only vocabulary selection differs;
  * parity set: FLORES+ dev (multi-parallel), pretokenized the same way;
  * to keep the pure-Python learner tractable on ~500 MB we drop pretoken types with count < min_count
    per language (rare types barely affect merge order; documented in the report).
The learned merges are wrapped into a Hugging Face byte-level BPE tokenizer with our pretokenizer.
"""
from __future__ import annotations

import importlib.util
import io
import json
import sys
from collections import Counter
from pathlib import Path

from tokenizers import Tokenizer, decoders, models
from tokenizers.pre_tokenizers import ByteLevel

from motok.baselines.hf import _pretok
from motok.pretokenize import count_pretokens, iter_lines

EXT = Path(__file__).resolve().parents[2] / "ext" / "parity-aware-bpe" / "parity_aware_bpe" / "parity_aware_learn_bpe.py"

_BL = ByteLevel(add_prefix_space=False, use_regex=False)


def to_bytelevel(s: str) -> str:
    return "".join(p for p, _ in _BL.pre_tokenize_str(s))


def _load_ext():
    spec = importlib.util.spec_from_file_location("parity_aware_learn_bpe", EXT)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["parity_aware_learn_bpe"] = mod
    spec.loader.exec_module(mod)
    return mod


def _dict_file(counter: Counter, min_count: int) -> io.StringIO:
    lines = [f"{to_bytelevel(p)} {c}" for p, c in counter.most_common() if c >= min_count]
    f = io.StringIO("\n".join(lines) + "\n")
    f.name = "<dict>"
    return f


def train_parity_bpe(lang_counts: dict[str, Counter], dev_files: dict[str, str], vocab_size: int, out_path: str,
                     variant: str = "base", min_count: int = 3, global_merges: int = 0, log=print) -> Tokenizer:
    mod = _load_ext()
    langs = list(lang_counts)
    inputs = [_dict_file(lang_counts[l], min_count) for l in langs]
    devs = [_dict_file(count_pretokens(iter_lines(dev_files[l]))[0], 1) for l in langs]
    n_merges = vocab_size - 256
    merges_io = io.StringIO()
    merges_io.name = "<merges>"
    log(f"[parity-bpe] {variant}: {n_merges} merges, types per lang: "
        + ", ".join(f"{l}={sum(1 for c in lang_counts[l].values() if c >= min_count)}" for l in langs))
    if variant == "base":
        mod.learn_bpe(inputs, merges_io, devs, n_merges, min_frequency=2, is_dict=True, num_global=global_merges)
    else:
        mod.learn_bpe_moving_window(inputs, merges_io, devs, n_merges, min_frequency=2, is_dict=True,
                                    num_global=global_merges)
    merges = [tuple(l.split(" ")) for l in merges_io.getvalue().splitlines()[1:] if l.strip()]
    tok = merges_to_tokenizer(merges)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    tok.save(out_path)
    return tok


def merges_to_tokenizer(merges: list[tuple[str, str]]) -> Tokenizer:
    vocab = {ch: i for i, ch in enumerate(ByteLevel.alphabet())}
    clean = []
    for a, b in merges:
        if a in vocab and b in vocab and a + b not in vocab:
            vocab[a + b] = len(vocab)
            clean.append((a, b))
    tok = Tokenizer(models.BPE(vocab=vocab, merges=clean))
    tok.pre_tokenizer = _pretok()
    tok.decoder = decoders.ByteLevel()
    return tok
