#!/usr/bin/env python3
"""Count the learned parameters of a trained HACT run (decoder + transition model, encoder excluded)."""
import glob, sys, torch
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
def walk(obj, prefix=""):
    """Yield (name, tensor) for every tensor anywhere inside nested dicts/lists."""
    if hasattr(obj, "numel"): yield prefix, obj
    elif isinstance(obj, dict):
        for k, v in obj.items(): yield from walk(v, f"{prefix}.{k}" if prefix else str(k))
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj): yield from walk(v, f"{prefix}[{i}]")


def count(path):
    obj = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(obj, dict): print("   top-level keys:", [(k, type(v).__name__) for k, v in list(obj.items())[:12]])
    tensors = dict(walk(obj))
    by_group = {}
    for k, v in tensors.items():
        g = k.split(".")[0]; by_group[g] = by_group.get(g, 0) + v.numel()
    return sum(v.numel() for v in tensors.values()), f"{len(tensors)} tensors; by top group {sorted(by_group.items(), key=lambda x: -x[1])[:8]}"


for label, pattern in (("transition model (hact/seed_0/fold_0)", "outputs/hact_reassembly_a/hact/seed_0/fold_0/*.pt"),
                       ("event decoder (_shared_observation/seed_0/fold_0)", "outputs/hact_reassembly_a/_shared_observation/seed_0/fold_0/*.pt")):
    for f in sorted(glob.glob(str(ROOT / pattern))):
        n, info = count(f); print(label, Path(f).name, n, info)
