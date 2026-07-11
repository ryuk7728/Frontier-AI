"""Shared experiment utilities."""

import json
import math
import os
import random
from pathlib import Path

import numpy as np
import torch


def psnr(mse: float) -> float:
    # Cap exact/numerical-zero reconstruction at 120 dB so plots remain finite.
    return -10.0 * math.log10(max(mse, 1e-12))


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def save_json(data, path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2)
