#!/usr/bin/env python3
"""Apply trained frame-level baseline fold models to a transfer benchmark.

For every fold, the model trained by ``train_frame_baselines.py`` on the source
benchmark is loaded unchanged and scores the test videos of the transfer
benchmark (whose train/val lists are the source fold lists).  Validation scores
are copied from the source run, so the threshold selection of
``evaluate_fully_predicted.py --decision validation_f1`` stays the one made on
the source validation participants.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate_fully_predicted import _read  # noqa: E402
from train_frame_baselines import CausalTCN, MistSenseRGB, load_video  # noqa: E402


def score_tcn(model, videos, ids, device):
    out = {}
    with torch.no_grad():
        for v in ids:
            x = torch.from_numpy(videos[v][0]).to(device).unsqueeze(0)
            out[v] = torch.sigmoid(model(x)).squeeze(0).cpu().numpy()
    return out


def score_mistsense(model, videos, ids, device, window, test_stride, batch_size=64):
    def batch_windows(items):
        xs = []
        for v, t in items:
            start = max(0, t - window + 1)
            chunk = videos[v][0][start : t + 1]
            if len(chunk) < window:
                chunk = np.concatenate([np.repeat(chunk[:1], window - len(chunk), axis=0), chunk])
            xs.append(chunk)
        return torch.from_numpy(np.stack(xs)).to(device)

    out = {}
    with torch.no_grad():
        for v in ids:
            length = len(videos[v][1])
            ends = list(range(0, length, test_stride))
            if ends[-1] != length - 1:
                ends.append(length - 1)
            logits = []
            for i in range(0, len(ends), batch_size):
                items = [(v, t) for t in ends[i : i + batch_size]]
                logits.append(torch.sigmoid(model(batch_windows(items))).cpu().numpy())
            values = np.concatenate(logits)
            scores = np.zeros(length, dtype=np.float32)
            for k, t in enumerate(ends):
                nxt = ends[k + 1] if k + 1 < len(ends) else length
                scores[t:nxt] = values[k]
            scores[: ends[0]] = values[0]
            out[v] = scores
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source-root", type=Path, required=True, help="train_frame_baselines output root")
    ap.add_argument("--benchmark", type=Path, required=True, help="transfer benchmark directory")
    ap.add_argument("--output-root", type=Path, required=True)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()
    manifest = _read(args.benchmark / "manifest.json")
    records = {str(r["video_id"]): r for r in manifest["workers"]}
    device = torch.device(args.device)
    for split in _read(args.benchmark / "splits.json")["folds"]:
        fold = int(split["fold"])
        source = args.source_root / f"fold_{fold}"
        meta = _read(source / "meta.json")
        if list(meta["training_video_ids"]) != list(split["train"]) or list(meta["validation_video_ids"]) != list(split["val"]):
            raise ValueError(f"fold {fold}: transfer benchmark train/val lists differ from the source model's")
        videos = {v: load_video(records[v]) for v in split["test"]}
        input_dim = next(iter(videos.values()))[0].shape[1]
        config = meta["config"]
        if meta["method"] == "causal_tcn":
            model = CausalTCN(input_dim, width=config["width"], dilations=config["dilations"], dropout=config["dropout"])
        else:
            model = MistSenseRGB(input_dim, width=config["width"], queries=config["queries"], layers=config["layers"])
        model.load_state_dict(torch.load(source / "model.pt", map_location="cpu"))
        model.to(device).eval()
        if meta["method"] == "causal_tcn":
            test_scores = score_tcn(model, videos, list(split["test"]), device)
        else:
            test_scores = score_mistsense(model, videos, list(split["test"]), device, config["window"], config["test_stride"])
        out = args.output_root / f"fold_{fold}"
        out.mkdir(parents=True, exist_ok=True)
        shutil.copy(source / "validation_scores.npz", out / "validation_scores.npz")
        np.savez_compressed(out / "test_scores.npz", **test_scores)
        (out / "meta.json").write_text(json.dumps({
            **{k: meta[k] for k in ("method", "fold", "seed", "config", "validation_video_ids", "training_video_ids", "parameters")},
            "test_video_ids": list(split["test"]), "source_model": str((source / "model.pt").resolve()),
            "transfer_benchmark": str(args.benchmark.resolve()), "test_annotations_consumed_by_model": False,
        }, indent=2) + "\n")
        print(f"[done] {meta['method']} fold={fold} test={len(test_scores)} videos")


if __name__ == "__main__":
    main()
