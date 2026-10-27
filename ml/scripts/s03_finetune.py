"""s03: fine-tune all-MiniLM-L6-v2 for spam detection (LP-FT: linear probe, then full fine-tune).

    uv run python -I scripts/s03_finetune.py [--frozen]

Inputs are norm_v1 `enc` strings (the same text the device encodes). Saves the best checkpoint by validation
PR-AUC to data/work/finetuned/ (HF model dir + metrics.json). --frozen stops after the linear probe, so the encoder
keeps its pretrained weights.
"""

from __future__ import annotations

import argparse
import copy
import json
import random
import sys
import time

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup

from smsml.augment import augment
from smsml.normalize import normalize
from smsml.pipeline import config, snapshot_dir, work


class SpamModel(nn.Module):
    def __init__(self, path: str):
        super().__init__()
        self.bert = AutoModel.from_pretrained(path)
        self.head = nn.Linear(self.bert.config.hidden_size, 1)

    def embed(self, ids: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        h = self.bert(input_ids=ids, attention_mask=mask).last_hidden_state
        m = mask.unsqueeze(-1).to(h.dtype)
        pooled = (h * m).sum(1) / m.sum(1).clamp(min=1e-9)
        return F.normalize(pooled, p=2, dim=-1)

    def forward(self, ids: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return self.head(self.embed(ids, mask).float()).squeeze(-1)


class Texts(Dataset):
    def __init__(self, texts: list[str], labels: np.ndarray):
        self.texts, self.labels = texts, labels

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, i: int):
        return self.texts[i], int(self.labels[i])


class Collate:
    def __init__(self, tok, max_len: int, train: bool, spam_p: float, ham_p: float, seed: int):
        self.tok, self.max_len, self.train = tok, max_len, train
        self.spam_p, self.ham_p = spam_p, ham_p
        self.r = random.Random(seed)

    def __call__(self, batch):
        texts = []
        for text, label in batch:
            if self.train and self.r.random() < (self.spam_p if label == 1 else self.ham_p):
                text = augment(text, self.r)
            texts.append(normalize(text).enc)
        enc = self.tok(texts, padding=True, truncation=True, max_length=self.max_len, return_tensors="pt")
        return enc["input_ids"], enc["attention_mask"], torch.tensor([b[1] for b in batch], dtype=torch.float32)


def worker_init(worker_id: int) -> None:
    info = torch.utils.data.get_worker_info()
    info.dataset  # noqa: B018 (each worker gets its own collate RNG via the seed below)
    random.seed(1000 + worker_id)


@torch.no_grad()
def evaluate(model: SpamModel, loader: DataLoader, device) -> float:
    model.eval()
    scores, labels = [], []
    for ids, mask, y in loader:
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            z = model(ids.to(device), mask.to(device))
        scores.append(z.float().cpu())
        labels.append(y)
    model.train()
    return float(average_precision_score(torch.cat(labels).numpy(), torch.cat(scores).numpy()))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--frozen", action="store_true")
    args = ap.parse_args()
    cfg = config()
    fc = cfg["finetune"]
    seed = int(cfg["seed"])
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    path = str(snapshot_dir(cfg))

    df = pd.read_parquet(work("corpus_split.parquet"))
    tr = df[df["split"] == "train"]
    va = df[df["split"].isin(["val_a", "val_b"])]
    n_spam, n_ham = int(tr["label"].sum()), int((tr["label"] == 0).sum())
    print(f"train: {len(tr)} ({n_spam} spam / {n_ham} ham); val: {len(va)}; device {device}")

    tok = AutoTokenizer.from_pretrained(path)
    model = SpamModel(path).to(device)
    max_len = int(cfg["model"]["max_len"])
    train_ds = Texts(tr["text"].tolist(), tr["label"].to_numpy())
    val_ds = Texts(va["text"].tolist(), va["label"].to_numpy())
    collate_train = Collate(tok, max_len, True, fc["aug_spam_p"], fc["aug_ham_p"], seed)
    collate_eval = Collate(tok, max_len, False, 0, 0, seed)
    g = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(train_ds, batch_size=fc["batch"], shuffle=True, collate_fn=collate_train,
                              num_workers=6, generator=g, persistent_workers=True, worker_init_fn=worker_init)
    val_loader = DataLoader(val_ds, batch_size=256, shuffle=False, collate_fn=collate_eval, num_workers=6)
    pos_weight = torch.tensor([n_ham / max(1, n_spam)], device=device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    history = []

    # Stage 1: linear probe (encoder frozen, no augmentation needed but harmless).
    for p in model.bert.parameters():
        p.requires_grad_(False)
    opt = torch.optim.AdamW(model.head.parameters(), lr=fc["lp_lr"])
    for epoch in range(fc["lp_epochs"]):
        t0 = time.time()
        for ids, mask, y in train_loader:
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                z = model(ids.to(device), mask.to(device))
            loss = loss_fn(z.float(), y.to(device))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        ap_val = evaluate(model, val_loader, device)
        history.append({"stage": "lp", "epoch": epoch, "val_pr_auc": ap_val, "secs": round(time.time() - t0, 1)})
        print(history[-1], flush=True)
    best = {"val_pr_auc": history[-1]["val_pr_auc"], "state": copy.deepcopy(model.state_dict()), "stage": "lp"}

    if not args.frozen:
        for p in model.bert.parameters():
            p.requires_grad_(True)
        decay = [p for n, p in model.bert.named_parameters() if not any(k in n for k in ("bias", "LayerNorm"))]
        no_decay = [p for n, p in model.bert.named_parameters() if any(k in n for k in ("bias", "LayerNorm"))]
        opt = torch.optim.AdamW([
            {"params": decay, "lr": fc["lr"], "weight_decay": fc["weight_decay"]},
            {"params": no_decay, "lr": fc["lr"], "weight_decay": 0.0},
            {"params": model.head.parameters(), "lr": fc["head_lr"], "weight_decay": 0.0},
        ])
        total = fc["ft_epochs"] * len(train_loader)
        sched = get_linear_schedule_with_warmup(opt, int(fc["warmup"] * total), total)
        step = 0
        for epoch in range(fc["ft_epochs"]):
            t0 = time.time()
            for ids, mask, y in train_loader:
                with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                    z = model(ids.to(device), mask.to(device))
                loss = loss_fn(z.float(), y.to(device))
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), fc["clip"])
                opt.step()
                sched.step()
                step += 1
                if step % fc["eval_every"] == 0 or step == total:
                    ap_val = evaluate(model, val_loader, device)
                    history.append({"stage": "ft", "epoch": epoch, "step": step, "loss": round(float(loss), 4),
                                    "val_pr_auc": ap_val, "secs": round(time.time() - t0, 1)})
                    print(history[-1], flush=True)
                    if ap_val > best["val_pr_auc"]:
                        best = {"val_pr_auc": ap_val, "state": copy.deepcopy(model.state_dict()), "stage": f"ft@{step}"}

    model.load_state_dict(best["state"])
    out = work("finetuned", "bert")
    out.mkdir(parents=True, exist_ok=True)
    model.bert.save_pretrained(out)
    tok.save_pretrained(out)
    work("finetuned", "metrics.json").write_text(json.dumps(
        {"best": {"val_pr_auc": best["val_pr_auc"], "stage": best["stage"]}, "frozen": args.frozen, "history": history},
        indent=2), encoding="utf-8")
    print(f"best {best['stage']} val PR-AUC {best['val_pr_auc']:.5f}; saved to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
