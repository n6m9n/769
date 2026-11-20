"""Load any trained tokenizer file (HF json, MorphBPE json, split-tree pkl) behind one interface."""
from __future__ import annotations

from pathlib import Path


def is_tokenizer_file(p: Path) -> bool:
    p = Path(p)
    if p.name.endswith((".vocab.json", ".morf.pkl")):
        return False
    return p.suffix in (".json", ".pkl")


def load_any(p):
    """Returns a wrapper with .encode/.encode_with_offsets/.decode_tokens/.vocab_size, or None."""
    p = Path(p)
    if not is_tokenizer_file(p):
        return None
    if p.name.startswith("morphbpe"):
        from motok.baselines.morph_bpe import MorphBPEWrapper
        return MorphBPEWrapper.load(p, p.parent / "morphbpe.morf.pkl", p.stem)
    if p.suffix == ".json":
        from motok.baselines.hf import HFWrapper
        return HFWrapper.load(p, p.stem)
    from motok.baselines.hf import SplitTreeWrapper
    from motok.toast.io import load_tokenizer
    return SplitTreeWrapper(load_tokenizer(p), p.stem)
