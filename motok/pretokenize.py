"""Pretokenization and pretoken counting.

Uses the length-limited GPT-4o regex from ToaST (Schmidt et al. 2026, Listing 2).
The letter classes include \\p{M}, so Indic vowel signs / viramas stay attached
to their consonants (GPT-2's regex would split them off).
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Iterable

import regex

GPT4O_REGEX_16 = "|".join([
    r"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]{0,32}[\p{Ll}\p{Lm}\p{Lo}\p{M}]{1,32}(?i:'s|'t|'re|'ve|'m|'ll|'d)?",
    r"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]{1,32}[\p{Ll}\p{Lm}\p{Lo}\p{M}]{0,32}(?i:'s|'t|'re|'ve|'m|'ll|'d)?",
    r"\p{N}{1,3}",
    r" ?[^\s\p{L}\p{N}]{1,16}[\r\n/]{0,16}",
    r"\s{0,15}[\r\n]{1,16}",
    r"\s{1,16}(?!\S)",
    r"\s{1,16}",
])

_PAT = regex.compile(GPT4O_REGEX_16)


def pretokenize(text: str) -> list[str]:
    return _PAT.findall(text)


def pretokenize_with_offsets(text: str) -> list[tuple[str, int]]:
    return [(m.group(0), m.start()) for m in _PAT.finditer(text)]


def iter_lines(path: str | Path, max_lines: int | None = None) -> Iterable[str]:
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if max_lines is not None and i >= max_lines:
                break
            line = line.rstrip("\n")
            if line:
                yield line


def count_pretokens(lines: Iterable[str]) -> tuple[Counter, dict]:
    """Count pretokens in a stream of lines. Returns (counter, stats)."""
    c: Counter = Counter()
    n_chars = n_bytes = n_lines = n_words = 0
    for line in lines:
        # count each line followed by a newline so whitespace stats are realistic
        c.update(pretokenize(line))
        n_chars += len(line)
        n_bytes += len(line.encode("utf-8"))
        n_words += len(line.split())
        n_lines += 1
    return c, {"chars": n_chars, "bytes": n_bytes, "lines": n_lines, "words": n_words}


def save_counts(path: str | Path, counter: Counter, stats: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"stats": stats, "counts": dict(counter.most_common())}, f, ensure_ascii=False)


def load_counts(path: str | Path) -> tuple[Counter, dict]:
    with open(path, encoding="utf-8") as f:
        d = json.load(f)
    return Counter(d["counts"]), d["stats"]


# ---------------------------------------------------------------- parallel counting
def _count_chunk(lines: list[str]) -> tuple[Counter, dict]:
    return count_pretokens(lines)


def count_pretokens_file(path: str | Path, max_lines: int | None = None, n_proc: int = 1,
                         chunk_lines: int = 20000) -> tuple[Counter, dict]:
    """count_pretokens over a file, split into line chunks counted in n_proc worker processes."""
    if n_proc <= 1:
        return count_pretokens(iter_lines(path, max_lines))
    from multiprocessing import Pool

    def chunks():
        buf = []
        for line in iter_lines(path, max_lines):
            buf.append(line)
            if len(buf) >= chunk_lines:
                yield buf
                buf = []
        if buf:
            yield buf

    total: Counter = Counter()
    stats = {"chars": 0, "bytes": 0, "lines": 0, "words": 0}
    with Pool(n_proc) as pool:
        for c, st in pool.imap_unordered(_count_chunk, chunks()):
            total.update(c)
            for k in stats:
                stats[k] += st[k]
    return total, stats
