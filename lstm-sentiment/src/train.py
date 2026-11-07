from __future__ import annotations

import json
from pathlib import Path

import torch

from src.model import train

ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT / "models"


def save_artifact(model) -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    path = MODEL_DIR / "model.pt"
    if isinstance(model, dict):
        payload = {key: value.state_dict() if hasattr(value, "state_dict") else value for key, value in model.items()}
        torch.save(payload, path)
    elif hasattr(model, "state_dict"):
        torch.save(model.state_dict(), path)
    else:
        torch.save(model, path)


def main() -> None:
    model, metrics = train()
    save_artifact(model)
    (MODEL_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
