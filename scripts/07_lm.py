#!/usr/bin/env python3
"""Downstream evaluation: train a small GPT per tokenizer on the same bytes, report bits per byte.

  python scripts/07_lm.py --config configs/full.yaml --tokenizer toast_compression --gpu 0

Fairness controls (RQ4):
  * same training text for every tokenizer: an equal byte budget per language (--train-mb, default 24 MB,
    bounded by the smallest corpus), lines shuffled with a fixed seed, one epoch;
  * same model (6 layers, d=512, 8 heads, ctx 512 tokens, tied embeddings; only the embedding table
    depends on the vocabulary, which is the same size for every tokenizer) and the same optimiser;
  * same tokens per step, so tokenizers that compress better see the data in fewer steps.
  * metric: bits per byte = total NLL / ln 2 / UTF-8 bytes, which is tokenization-independent.
    Held-out: FineWeb(-2) eval split (non-parallel) and FLORES+ devtest (parallel: same content).

Encoded corpora are cached in <results>/lm/data/<tokenizer>/.
"""
import argparse, json, math, os, random, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from motok.config import load_cfg, read_lines
from motok.tokload import load_any

ap = argparse.ArgumentParser()
ap.add_argument("--config", required=True)
ap.add_argument("--tokenizer", required=True)
ap.add_argument("--gpu", type=int, default=0)
ap.add_argument("--train-mb", type=float, default=24.0, help="MB of training text per language")
ap.add_argument("--eval-lines", type=int, default=3000)
ap.add_argument("--procs", type=int, default=24)
ap.add_argument("--ctx", type=int, default=512)
ap.add_argument("--batch", type=int, default=64)
ap.add_argument("--layers", type=int, default=6)
ap.add_argument("--dim", type=int, default=512)
ap.add_argument("--heads", type=int, default=8)
ap.add_argument("--lr", type=float, default=1.5e-3)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--encode-only", action="store_true")
a = ap.parse_args()
cfg = load_cfg(a.config)
tok_path = next(p for p in [cfg.out / "tokenizers" / f"{a.tokenizer}{s}" for s in (".pkl", ".json")] if p.exists())
lmdir = cfg.out / "lm"
ddir = lmdir / "data" / a.tokenizer
ddir.mkdir(parents=True, exist_ok=True)


# ------------------------------------------------------------------ encoding (CPU, parallel)
_TOK = None
_IDS = None


def _init(path):
    global _TOK, _IDS
    _TOK = load_any(path)
    if hasattr(_TOK, "st"):  # split-tree: token string -> id (vocab, then 256 byte tokens)
        _IDS = {t: i for i, t in enumerate(list(_TOK.st.vocab) + _TOK.st.byte_tokens)}


def _encode(lines):
    out = []
    for t in lines:
        if _IDS is not None:
            out.append(np.array([_IDS[x] for x in _TOK.encode(t)], dtype=np.int32))
        elif hasattr(_TOK, "encode_ids"):
            out.append(np.array(_TOK.encode_ids(t), dtype=np.int32))
        else:
            out.append(np.array(_TOK.tok.encode(t, add_special_tokens=False).ids, dtype=np.int32))
    return out


def encode_lines(lines, pool, chunk=2000):
    chunks = [lines[i:i + chunk] for i in range(0, len(lines), chunk)]
    res = []
    for r in pool.imap(_encode, chunks):
        res.extend(r)
    return res


def take_mb(path, mb, from_end=True):
    lines = read_lines(path)
    if from_end:
        lines = lines[::-1]
    out, n = [], 0
    for l in lines:
        out.append(l); n += len(l.encode("utf-8"))
        if n >= mb * 1e6:
            break
    return out


def build_data():
    meta_p = ddir / "meta.json"
    if meta_p.exists():
        return json.load(open(meta_p))
    from multiprocessing import Pool
    t0 = time.time()
    tok = load_any(tok_path)
    V = tok.vocab_size
    eos = V
    rng = random.Random(1234)   # same order for every tokenizer
    train = []
    for l in cfg.langs:
        train += take_mb(cfg.data / "raw" / f"{l}.txt", a.train_mb)
    rng.shuffle(train)
    meta = {"vocab": V, "eos": eos, "train_bytes": sum(len(x.encode()) for x in train), "eval": {}}
    with Pool(a.procs, initializer=_init, initargs=(str(tok_path),)) as pool:
        ids = encode_lines(train, pool)
        arr = np.concatenate([np.r_[x, eos] for x in ids]).astype(np.int32)
        arr.tofile(ddir / "train.i32")
        meta["train_tokens"] = int(len(arr))
        for split, src in [("fw", cfg.data / "eval"), ("flores", cfg.data / "flores" / "devtest")]:
            for l in cfg.langs:
                p = src / f"{l}.txt"
                if not p.exists():
                    continue
                lines = read_lines(p, a.eval_lines)
                e = encode_lines(lines, pool)
                arr = np.concatenate([np.r_[x, eos] for x in e]).astype(np.int32)
                np.save(ddir / f"{split}_{l}.npy", arr)
                meta["eval"][f"{split}_{l}"] = {"bytes": sum(len(x.encode()) for x in lines), "tokens": int(len(arr))}
    meta["encode_seconds"] = time.time() - t0
    json.dump(meta, open(meta_p, "w"), indent=1)
    return meta


meta = build_data()
print(f"[lm] {a.tokenizer}: {meta['train_tokens']/1e6:.1f}M train tokens for {meta['train_bytes']/1e6:.0f} MB "
      f"({meta['train_bytes']/meta['train_tokens']:.2f} bytes/token)", flush=True)
if a.encode_only:
    sys.exit(0)

# ------------------------------------------------------------------ model (GPU)
os.environ["CUDA_VISIBLE_DEVICES"] = str(a.gpu)
import torch
import torch.nn as nn
import torch.nn.functional as F

dev = torch.device("cuda", 0)
torch.manual_seed(a.seed); np.random.seed(a.seed)
torch.backends.cuda.matmul.allow_tf32 = True


class Block(nn.Module):
    def __init__(self, d, h):
        super().__init__()
        self.h = h
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.qkv, self.proj = nn.Linear(d, 3 * d, bias=False), nn.Linear(d, d, bias=False)
        self.fc, self.out = nn.Linear(d, 4 * d, bias=False), nn.Linear(4 * d, d, bias=False)

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.qkv(self.ln1(x)).view(B, T, 3, self.h, C // self.h).permute(2, 0, 3, 1, 4)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        x = x + self.proj(y.transpose(1, 2).reshape(B, T, C))
        return x + self.out(F.gelu(self.fc(self.ln2(x))))


class GPT(nn.Module):
    def __init__(self, V, ctx, L, d, h):
        super().__init__()
        self.emb, self.pos = nn.Embedding(V, d), nn.Embedding(ctx, d)
        self.blocks = nn.ModuleList(Block(d, h) for _ in range(L))
        self.ln = nn.LayerNorm(d)
        self.apply(self._init)
        for n, p in self.named_parameters():
            if n.endswith(("proj.weight", "out.weight")):
                nn.init.normal_(p, std=0.02 / math.sqrt(2 * L))

    @staticmethod
    def _init(m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, std=0.02)

    def forward(self, idx):
        x = self.emb(idx) + self.pos(torch.arange(idx.shape[1], device=idx.device))
        for b in self.blocks:
            x = b(x)
        return self.ln(x) @ self.emb.weight.T   # tied output embedding


V = meta["vocab"] + 1
model = GPT(V, a.ctx, a.layers, a.dim, a.heads).to(dev)
n_params = sum(p.numel() for p in model.parameters())
n_nonemb = n_params - model.emb.weight.numel() - model.pos.weight.numel()
data = torch.from_numpy(np.fromfile(ddir / "train.i32", dtype=np.int32).astype(np.int64))
n_seq = (len(data) - 1) // a.ctx
steps = n_seq // a.batch
decay = [p for n, p in model.named_parameters() if p.dim() >= 2]
no_decay = [p for n, p in model.named_parameters() if p.dim() < 2]
opt = torch.optim.AdamW([{"params": decay, "weight_decay": 0.1}, {"params": no_decay, "weight_decay": 0.0}],
                        lr=a.lr, betas=(0.9, 0.95), fused=True)
warm = max(50, steps // 50)
sched = torch.optim.lr_scheduler.LambdaLR(
    opt, lambda s: min(1.0, (s + 1) / warm) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(1.0, s / steps)))))
print(f"[lm] params {n_params/1e6:.1f}M ({n_nonemb/1e6:.1f}M non-embedding), {steps} steps x {a.batch}x{a.ctx} tokens",
      flush=True)

perm = torch.randperm(n_seq, generator=torch.Generator().manual_seed(a.seed))
t0 = time.time()
log = []
model.train()
for s in range(steps):
    idx = perm[s * a.batch:(s + 1) * a.batch]
    starts = idx * a.ctx
    rows = torch.stack([data[i:i + a.ctx + 1] for i in starts.tolist()]).to(dev, non_blocking=True)
    x, y = rows[:, :-1], rows[:, 1:]
    with torch.autocast("cuda", dtype=torch.bfloat16):
        loss = F.cross_entropy(model(x).float().view(-1, V), y.reshape(-1))
    opt.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step(); sched.step()
    if s % 100 == 0 or s == steps - 1:
        tps = (s + 1) * a.batch * a.ctx / (time.time() - t0)
        log.append({"step": s, "loss": float(loss), "lr": sched.get_last_lr()[0]})
        print(f"  step {s:5d}/{steps} loss {float(loss):.4f} ({tps/1e3:.0f}k tok/s)", flush=True)
train_seconds = time.time() - t0


@torch.no_grad()
def bpb(arr, nbytes, eos, stride=None):
    """Sliding-window NLL: each window has ctx tokens, only the last `stride` targets are scored
    (all targets in the first window), so every token sees >= ctx-stride tokens of context."""
    model.eval()
    stride = stride or a.ctx // 2
    t = torch.from_numpy(arr.astype(np.int64))
    nll, n_scored = 0.0, 0
    starts = list(range(0, max(1, len(t) - 1 - a.ctx), stride)) + [max(0, len(t) - 1 - a.ctx)]
    starts = sorted(set(starts))
    prev_end = 0
    for i in range(0, len(starts), 64):
        batch = starts[i:i + 64]
        rows = torch.stack([t[s:s + a.ctx + 1] for s in batch]).to(dev)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(rows[:, :-1]).float()
        lp = -F.cross_entropy(logits.reshape(-1, V), rows[:, 1:].reshape(-1), reduction="none").view(len(batch), -1)
        for r, s in enumerate(batch):
            # score targets at absolute positions (s+1 .. s+ctx) not scored yet
            first = max(prev_end, s + 1)
            if first > s + a.ctx:
                continue
            tgt = rows[r, 1:]
            sl = slice(first - s - 1, a.ctx)
            mask = tgt[sl] != eos
            nll -= float(lp[r, sl][mask].sum()); n_scored += int((tgt[sl] != eos).sum())
            prev_end = s + a.ctx + 1
    return nll / math.log(2) / nbytes, n_scored


res = {"tokenizer": a.tokenizer, "params": n_params, "params_nonemb": n_nonemb, "steps": steps,
       "train_tokens": meta["train_tokens"], "train_bytes": meta["train_bytes"], "train_seconds": train_seconds,
       "final_loss": log[-1]["loss"], "log": log, "args": vars(a), "bpb": {}}
for key, m in meta["eval"].items():
    arr = np.load(ddir / f"{key}.npy")
    b, n = bpb(arr, m["bytes"], meta["eos"])
    res["bpb"][key] = b
    print(f"  {key:<12} bpb {b:.4f}  ({n} tokens, {m['bytes']} bytes)", flush=True)
for split in ["fw", "flores"]:
    vals = [v for k, v in res["bpb"].items() if k.startswith(split + "_")]
    if vals:
        res[f"{split}_bpb_mean"], res[f"{split}_bpb_max"] = float(np.mean(vals)), float(np.max(vals))
out = lmdir / f"{a.tokenizer}_s{a.seed}.json"
json.dump(res, open(out, "w"), indent=1)
print(f"[lm] done in {train_seconds/60:.1f} min -> {out}", flush=True)
