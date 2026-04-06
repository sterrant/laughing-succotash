#!/usr/bin/env python3
"""Train a tiny MLP policy from logged turn data (behavior cloning)."""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path
from typing import List, Tuple

import torch
import torch.nn as nn


FEATURES = ["altitude", "velocity", "fuel", "sec"]
BURN_MIN = 0.0
BURN_MAX = 30.0
BURN_STEP = 0.1
NUM_CLASSES = int(round((BURN_MAX - BURN_MIN) / BURN_STEP)) + 1


class TinyMLP(nn.Module):
    def __init__(self, in_dim: int = 4, hidden: int = 32, out_dim: int = NUM_CLASSES):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x):
        return self.net(x)


def burn_to_class(burn: float) -> int:
    clamped = max(BURN_MIN, min(BURN_MAX, float(burn)))
    return int(round((clamped - BURN_MIN) / BURN_STEP))


def class_to_burn(cls: int) -> float:
    cls = max(0, min(NUM_CLASSES - 1, int(cls)))
    return BURN_MIN + (float(cls) * BURN_STEP)


def load_rows(path: Path) -> List[Tuple[List[float], int]]:
    rows: List[Tuple[List[float], int]] = []
    with path.open("r", encoding="utf-8", newline="") as fp:
        reader = csv.DictReader(fp)
        for r in reader:
            try:
                x = [float(r[f]) for f in FEATURES]
                y = burn_to_class(float(r["burn"]))
            except (ValueError, KeyError, TypeError):
                continue
            if y < 0 or y >= NUM_CLASSES:
                continue
            rows.append((x, y))
    if not rows:
        raise SystemExit(f"No valid training rows found in {path}")
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description="Train tiny MLP policy from turns.csv")
    ap.add_argument("--turn-csv", default="logs/turns.csv")
    ap.add_argument("--out-model", default="models/policy.pt")
    ap.add_argument("--out-norm", default="models/policy_norm.json")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--hidden", type=int, default=32)
    ap.add_argument("--seed", type=int, default=1337)
    args = ap.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    rows = load_rows(Path(args.turn_csv))
    random.shuffle(rows)
    split = max(1, int(len(rows) * 0.8))
    train_rows = rows[:split]
    val_rows = rows[split:] if split < len(rows) else rows[-1:]

    x_train = torch.tensor([x for x, _ in train_rows], dtype=torch.float32)
    y_train = torch.tensor([y for _, y in train_rows], dtype=torch.long)
    x_val = torch.tensor([x for x, _ in val_rows], dtype=torch.float32)
    y_val = torch.tensor([y for _, y in val_rows], dtype=torch.long)

    mean = x_train.mean(dim=0)
    std = x_train.std(dim=0)
    std[std < 1e-6] = 1.0

    x_train_n = (x_train - mean) / std
    x_val_n = (x_val - mean) / std

    model = TinyMLP(in_dim=len(FEATURES), hidden=args.hidden, out_dim=NUM_CLASSES)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(1, args.epochs + 1):
        model.train()
        logits = model(x_train_n)
        loss = loss_fn(logits, y_train)
        opt.zero_grad()
        loss.backward()
        opt.step()

        model.eval()
        with torch.no_grad():
            val_logits = model(x_val_n)
            val_loss = loss_fn(val_logits, y_val).item()
            val_pred = torch.argmax(val_logits, dim=1)
            val_acc = (val_pred == y_val).float().mean().item()

        print(
            f"epoch={epoch:03d} train_loss={loss.item():.4f} "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.3f}"
        )

    out_model = Path(args.out_model)
    out_model.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "num_classes": NUM_CLASSES,
            "hidden": args.hidden,
            "features": FEATURES,
            "burn_step": BURN_STEP,
            "burn_min": BURN_MIN,
            "burn_max": BURN_MAX,
        },
        out_model,
    )

    out_norm = Path(args.out_norm)
    out_norm.parent.mkdir(parents=True, exist_ok=True)
    out_norm.write_text(
        json.dumps({"mean": mean.tolist(), "std": std.tolist(), "features": FEATURES}, indent=2),
        encoding="utf-8",
    )
    print(f"saved model: {out_model}")
    print(f"saved norm:  {out_norm}")


if __name__ == "__main__":
    main()
