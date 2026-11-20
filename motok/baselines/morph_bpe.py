"""MorphBPE baseline (Asgari et al., Findings ACL 2026), reimplemented (no public code found).

MorphBPE = BPE whose merges never cross a morpheme boundary. We realise this exactly by inserting an
invisible separator (U+2063) at every morpheme boundary of every word before BPE training and
inference; the pre-tokenizer splits on it, so no token can span two morphemes.

Morphology source: one joint Morfessor Baseline model over the most frequent words of all languages
(language-agnostic at inference, like every other tokenizer here). It is the same kind of
unsupervised training signal the LP uses; gold morphology stays evaluation-only.
"""
from __future__ import annotations

from collections import Counter
from pathlib import Path

import regex
from tokenizers import Regex, Tokenizer, decoders, models, pre_tokenizers, trainers

from motok.morph.segment import MorfessorSegmenter, train_morfessor
from motok.pretokenize import GPT4O_REGEX_16, pretokenize, pretokenize_with_offsets
from motok.toast.trees import body_start

SEP = "⁣"
_WORD = regex.compile(r"^[\p{L}\p{M}]+$")


def _pretok_morph():
    return pre_tokenizers.Sequence([
        pre_tokenizers.Split(Regex(GPT4O_REGEX_16), behavior="isolated"),
        pre_tokenizers.Split(SEP, behavior="removed"),
        pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False),
    ])


def train_joint_morfessor(word_counts: dict[str, Counter], words_per_lang: int = 20000):
    joint: Counter = Counter()
    for l, wc in word_counts.items():
        top = [(w, c) for w, c in wc.most_common() if _WORD.match(w)][:words_per_lang]
        joint.update(dict(top))
    return train_morfessor(joint, max_words=len(joint))


class MorphMarker:
    """Insert SEP at morpheme boundaries inside each pretoken body (cached per pretoken)."""

    def __init__(self, segmenter):
        self.seg = segmenter
        self._cache: dict[str, str] = {}

    def mark_pretoken(self, p: str) -> str:
        hit = self._cache.get(p)
        if hit is not None:
            return hit
        b0 = body_start(p)
        body = p[b0:]
        out = p
        segs = self.seg(body) if body and _WORD.match(body) else None
        if segs and len(segs) > 1 and "".join(segs) == body:
            out = p[:b0] + SEP.join(segs)
        if len(self._cache) < 5_000_000:
            self._cache[p] = out
        return out

    def mark(self, text: str) -> str:
        return "".join(self.mark_pretoken(p) for p in pretokenize(text))


def train_morph_bpe(lines, marker: MorphMarker, vocab_size: int, save_path: str) -> Tokenizer:
    tok = Tokenizer(models.BPE())
    trainer = trainers.BpeTrainer(vocab_size=vocab_size, initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
                                  show_progress=False)
    tok.pre_tokenizer = _pretok_morph()
    tok.decoder = decoders.ByteLevel()
    tok.train_from_iterator((marker.mark(l) for l in lines), trainer=trainer)
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    tok.save(save_path)
    return tok


class MorphBPEWrapper:
    """Evaluation interface: segment -> BPE within morphemes; offsets mapped back to the original text."""

    def __init__(self, tok: Tokenizer, marker: MorphMarker, name="morphbpe"):
        self.tok, self.marker, self.name = tok, marker, name

    @classmethod
    def load(cls, tok_path, morf_path, name=None):
        return cls(Tokenizer.from_file(str(tok_path)), MorphMarker(MorfessorSegmenter.load(morf_path)),
                   name or Path(tok_path).stem)

    @property
    def vocab_size(self):
        return self.tok.get_vocab_size()

    def _marked(self, text):
        out, pos_map = [], []
        for p, o in pretokenize_with_offsets(text):
            m = self.marker.mark_pretoken(p)
            k = 0
            for ch in m:
                if ch == SEP:
                    pos_map.append(o + k)    # separator: maps to the boundary position
                else:
                    pos_map.append(o + k); k += 1
            out.append(m)
        pos_map.append(len(text))
        return "".join(out), pos_map

    def encode(self, text):
        m, _ = self._marked(text)
        return self.tok.encode(m, add_special_tokens=False).tokens

    def encode_ids(self, text):
        m, _ = self._marked(text)
        return self.tok.encode(m, add_special_tokens=False).ids

    def encode_with_offsets(self, text):
        m, pm = self._marked(text)
        e = self.tok.encode(m, add_special_tokens=False)
        return e.tokens, [(pm[s], pm[t] if t < len(pm) else len(text)) for s, t in e.offsets]

    def decode_tokens(self, text):
        return self.tok.decode(self.encode_ids(text))
