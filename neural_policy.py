#!/usr/bin/env python3
"""Neural-network policy for Lunar Lander (inference only)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn as nn


class TinyMLP(nn.Module):
    def __init__(self, in_dim: int = 4, hidden: int = 32, out_dim: int = 31):
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


class NeuralPolicy:
    """Torch-backed policy that predicts burn class from game state."""

    def __init__(self, cfg: Dict):
        policy_cfg = cfg["policy"]
        self.min_burn = int(policy_cfg.get("min_burn", 0))
        self.max_burn = int(policy_cfg.get("max_burn", 30))
        self.model_path = Path(policy_cfg.get("neural_model_path", "models/policy.pt"))
        self.norm_path = Path(policy_cfg.get("neural_norm_path", "models/policy_norm.json"))

        norm = json.loads(self.norm_path.read_text(encoding="utf-8"))
        self.mean = torch.tensor(norm["mean"], dtype=torch.float32)
        self.std = torch.tensor(norm["std"], dtype=torch.float32)

        ckpt = torch.load(self.model_path, map_location="cpu")
        self.model = TinyMLP(
            in_dim=4,
            hidden=int(ckpt.get("hidden", 32)),
            out_dim=int(ckpt.get("num_classes", 31)),
        )
        self.model.load_state_dict(ckpt["state_dict"])
        self.model.eval()

    def choose_burn(self, state: Optional[object]) -> int:
        if state is None:
            return 0

        alt = getattr(state, "altitude", None)
        vel = getattr(state, "velocity", None)
        fuel = getattr(state, "fuel", None)
        sec = getattr(state, "sec", None)

        if fuel is None or fuel <= 0:
            return 0

        # Use conservative defaults if a parsed field is unexpectedly missing.
        x = torch.tensor(
            [[
                float(alt if alt is not None else 9999.0),
                float(vel if vel is not None else 0.0),
                float(fuel),
                float(sec if sec is not None else 0.0),
            ]],
            dtype=torch.float32,
        )
        x = (x - self.mean) / self.std

        with torch.no_grad():
            logits = self.model(x)
            burn = int(torch.argmax(logits, dim=1).item())

        burn = max(self.min_burn, min(self.max_burn, burn))
        burn = min(burn, int(fuel))
        return burn

    def get_params(self) -> Dict:
        return {
            "policy_type": "neural",
            "model_path": str(self.model_path),
            "norm_path": str(self.norm_path),
            "min_burn": self.min_burn,
            "max_burn": self.max_burn,
        }

    def set_params(self, params: Dict) -> None:
        del params
