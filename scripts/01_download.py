#!/usr/bin/env python3
"""Download all data the pipeline needs into data/.

  python scripts/01_download.py --what smoke        # UD treebanks + SIGMORPHON (GitHub only, ~150MB, fast)
  python scripts/01_download.py --what morph        # SIGMORPHON 2022 gold segmentation
  python scripts/01_download.py --what fineweb      # training text per language (Hugging Face)
  python scripts/01_download.py --what flores       # FLORES+ dev/devtest parallel eval (gated: needs HF login)
  python scripts/01_download.py --what all

Layout produced:
  data/raw/{lang}.txt            training text, one paragraph per line
  data/eval/{lang}.txt           held-out text (non-parallel)
  data/flores/{split}/{lang}.txt parallel sentences, line-aligned across languages
  data/morph/{lang}.{split}.tsv  gold segmentations: word<TAB>m1 @@m2 ...
"""
from __future__ import annotations

import argparse
import random
import sys
import urllib.request
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
CFG = yaml.safe_load(open(ROOT / "configs" / "languages.yaml"))
LANGS = CFG["languages"]

UD = {  # smoke-test corpora hosted on GitHub
    "en": ("UD_English-EWT", "en_ewt"), "hi": ("UD_Hindi-HDTB", "hi_hdtb"),
    "mr": ("UD_Marathi-UFAL", "mr_ufal"), "ta": ("UD_Tamil-TTB", "ta_ttb"),
    "ru": ("UD_Russian-GSD", "ru_gsd"), "hu": ("UD_Hungarian-Szeged", "hu_szeged"),
}
SIGMORPHON = "https://raw.githubusercontent.com/sigmorphon/2022SegmentationST/main/data"


def fetch(url: str, dest: Path) -> bool:
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        urllib.request.urlretrieve(url, dest)
        return True
    except Exception as e:  # noqa: BLE001
        print(f"  ! failed {url}: {e}")
        return False


def download_morph():
    for lang, spec in LANGS.items():
        code = spec.get("sigmorphon")
        if not code:
            continue
        for split in ["train", "dev", "test.gold"]:
            out = DATA / "morph" / f"{lang}.{split.replace('.gold', '')}.tsv"
            if out.exists():
                continue
            print(f"morph {lang} {split}")
            fetch(f"{SIGMORPHON}/{code}.word.{split}.tsv", out)


def download_smoke(eval_frac=0.1, seed=0):
    """UD treebank sentences -> data/raw + data/eval. Small, but real text in every language."""
    rng = random.Random(seed)
    for lang in LANGS:
        if lang not in UD:
            print(f"  (no UD smoke corpus configured for {lang})")
            continue
        repo, stem = UD[lang]
        tmp = DATA / "ud" / f"{lang}.conllu"
        if not tmp.exists():
            ok = any(fetch(f"https://raw.githubusercontent.com/UniversalDependencies/{repo}/{br}/{stem}-ud-train.conllu", tmp)
                     for br in ["master", "dev"])
            if not ok:
                continue
        sents = [l[len("# text = "):].strip() for l in open(tmp, encoding="utf-8") if l.startswith("# text = ")]
        rng.shuffle(sents)
        n_eval = max(50, int(len(sents) * eval_frac))
        for name, part in [("eval", sents[:n_eval]), ("raw", sents[n_eval:])]:
            p = DATA / name / f"{lang}.txt"
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text("\n".join(part) + "\n", encoding="utf-8")
        print(f"smoke {lang}: {len(sents) - n_eval} train / {n_eval} eval sentences")
    download_morph()


def download_fineweb(eval_docs=2000):
    from datasets import load_dataset
    mb = CFG.get("download_mb", {})
    for lang, spec in LANGS.items():
        out, ev = DATA / "raw" / f"{lang}.txt", DATA / "eval" / f"{lang}.txt"
        if out.exists() and out.stat().st_size > 1e6:
            print(f"fineweb {lang}: exists, skipping")
            continue
        target = int(mb.get(lang, 50) * 1e6)
        fw = spec["fineweb"]
        print(f"fineweb {lang}: streaming {fw['repo']}/{fw['config']} -> {target/1e6:.0f} MB")
        ds = load_dataset(fw["repo"], name=fw["config"], split="train", streaming=True)
        out.parent.mkdir(parents=True, exist_ok=True); ev.parent.mkdir(parents=True, exist_ok=True)
        n_bytes = 0
        with open(out, "w", encoding="utf-8") as f_tr, open(ev, "w", encoding="utf-8") as f_ev:
            for i, ex in enumerate(ds):
                # one line per paragraph; first `eval_docs` documents are held out
                paras = [p.strip() for p in ex["text"].split("\n") if len(p.strip()) > 20]
                if i < eval_docs:
                    f_ev.write("\n".join(paras) + "\n")
                    continue
                block = "\n".join(paras) + "\n"
                f_tr.write(block)
                n_bytes += len(block.encode("utf-8"))
                if n_bytes >= target:
                    break
        print(f"  wrote {n_bytes/1e6:.1f} MB")


def download_flores():
    from datasets import load_dataset
    for split in ["dev", "devtest"]:
        for lang, spec in LANGS.items():
            out = DATA / "flores" / split / f"{lang}.txt"
            if out.exists():
                continue
            out.parent.mkdir(parents=True, exist_ok=True)
            try:
                ds = load_dataset("openlanguagedata/flores_plus", spec["flores"], split=split)
                ds = ds.sort("id")  # keep line alignment across languages
                out.write_text("\n".join(ds["text"]) + "\n", encoding="utf-8")
                print(f"flores {split} {lang}: {len(ds)} sentences")
            except Exception as e:  # noqa: BLE001
                print(f"  ! FLORES+ {lang} failed: {e}\n"
                      "    FLORES+ is gated: accept the terms at https://huggingface.co/datasets/openlanguagedata/flores_plus\n"
                      "    then run `huggingface-cli login`.")
                return


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--what", choices=["smoke", "morph", "fineweb", "flores", "all"], required=True)
    a = ap.parse_args()
    if a.what == "smoke":
        download_smoke()
    if a.what in ("morph", "all"):
        download_morph()
    if a.what in ("fineweb", "all"):
        download_fineweb()
    if a.what in ("flores", "all"):
        download_flores()
    sys.exit(0)
