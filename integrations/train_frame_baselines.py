#!/usr/bin/env python3
"""Supervised frame-level baselines on the shared cached features.

Two controlled adaptations that use no procedure representation:

* ``causal_tcn`` -- a dilated causal temporal convolutional network over the
  frame features (Lea et al., 2017), scoring every frame from its past.
* ``mistsense`` -- the RGB Video-Q-Former classification path of MistSense
  (Patsch et al., 2025) on the shared features: learned queries attend to a
  causal window ending at the current frame and a linear head scores it.

Both models are trained on the training participants with frame-level anomaly
targets (recovery frames are normal), selected on validation participants by
frame AUPRC, and produce dense per-frame scores for the validation and test
videos of one fold.  Outputs follow the layout read by
``evaluate_fully_predicted.py dense``.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate_fully_predicted import _average_precision, _ground_truth, _read  # noqa: E402


class CausalConvBlock(nn.Module):
    def __init__(self, width: int, dilation: int, dropout: float) -> None:
        super().__init__()
        self.left_padding = 2 * dilation
        self.conv = nn.Conv1d(width, width, kernel_size=3, dilation=dilation, padding=0)
        self.norm = nn.LayerNorm(width)
        self.dropout = nn.Dropout(dropout)

    def forward(self, value: Tensor) -> Tensor:
        residual = value
        value = F.pad(value, (self.left_padding, 0))
        value = self.conv(value).transpose(1, 2)
        value = self.dropout(F.gelu(self.norm(value))).transpose(1, 2)
        return residual + value


class CausalTCN(nn.Module):
    def __init__(self, input_dim: int, width: int = 256, dilations=(1, 2, 4, 8, 16, 32), dropout: float = 0.2) -> None:
        super().__init__()
        self.input = nn.Sequential(nn.Linear(input_dim, width), nn.LayerNorm(width), nn.GELU())
        self.blocks = nn.ModuleList(CausalConvBlock(width, int(d), dropout) for d in dilations)
        self.head = nn.Linear(width, 1)

    def forward(self, value: Tensor) -> Tensor:  # [B, T, D] -> [B, T]
        value = self.input(value).transpose(1, 2)
        for block in self.blocks:
            value = block(value)
        return self.head(value.transpose(1, 2)).squeeze(-1)


class MistSenseRGB(nn.Module):
    def __init__(self, input_dim: int, width: int = 384, queries: int = 32, layers: int = 2, dropout: float = 0.1) -> None:
        super().__init__()
        self.memory_projection = nn.Sequential(nn.Linear(input_dim, width), nn.LayerNorm(width))
        self.queries = nn.Parameter(torch.empty(queries, width))
        nn.init.normal_(self.queries, std=0.02)
        layer = nn.TransformerDecoderLayer(
            d_model=width, nhead=8, dim_feedforward=4 * width, dropout=dropout,
            activation="gelu", batch_first=True, norm_first=True,
        )
        self.qformer = nn.TransformerDecoder(layer, num_layers=layers)
        self.head = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, 1))

    def forward(self, value: Tensor) -> Tensor:  # [B, W, D] -> [B]
        memory = self.memory_projection(value)
        query = self.queries.unsqueeze(0).expand(len(value), -1, -1)
        representation = self.qformer(query, memory).mean(dim=1)
        return self.head(representation).squeeze(-1)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_video(record: dict) -> tuple[np.ndarray, np.ndarray]:
    with np.load(record["feature_path"]) as payload:
        features = np.asarray(payload["features"], dtype=np.float32)
    target, _ = _ground_truth(record)
    length = min(len(features), len(target))
    return features[:length], target[:length].astype(np.float32)


def frame_auprc(scores: dict[str, np.ndarray], targets: dict[str, np.ndarray]) -> float:
    ids = sorted(scores)
    return _average_precision(
        np.concatenate([targets[v][: len(scores[v])] for v in ids]).astype(np.int64),
        np.concatenate([scores[v] for v in ids]),
    )


def run_tcn(videos, split, device, seed, epochs, patience, lr):
    model = CausalTCN(next(iter(videos.values()))[0].shape[1]).to(device)
    positives = sum(float(videos[v][1].sum()) for v in split["train"])
    negatives = sum(float(len(videos[v][1]) - videos[v][1].sum()) for v in split["train"])
    pos_weight = torch.tensor(negatives / max(positives, 1.0), device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    def score_split(ids):
        model.eval()
        out = {}
        with torch.no_grad():
            for v in ids:
                x = torch.from_numpy(videos[v][0]).to(device).unsqueeze(0)
                out[v] = torch.sigmoid(model(x)).squeeze(0).cpu().numpy()
        return out

    best, best_state, bad, history = -1.0, None, 0, []
    order = list(split["train"])
    rng = random.Random(seed)
    for epoch in range(epochs):
        model.train()
        rng.shuffle(order)
        total = 0.0
        for v in order:
            x, y = videos[v]
            x = torch.from_numpy(x).to(device).unsqueeze(0)
            y = torch.from_numpy(y).to(device).unsqueeze(0)
            loss = F.binary_cross_entropy_with_logits(model(x), y, pos_weight=pos_weight)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total += float(loss)
        val = frame_auprc(score_split(split["val"]), {v: videos[v][1] for v in split["val"]})
        history.append({"epoch": epoch + 1, "train_loss": total / len(order), "val_auprc": val})
        if val > best:
            best, bad = val, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(best_state)
    return model, score_split(split["val"]), score_split(split["test"]), history, best


def run_mistsense(videos, split, device, seed, epochs, patience, lr, window, train_stride, test_stride, batch_size):
    model = MistSenseRGB(next(iter(videos.values()))[0].shape[1]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    # window ending at frame t covers frames [max(0, t-window+1), t]
    train_items = []
    for v in split["train"]:
        length = len(videos[v][1])
        train_items.extend((v, t) for t in range(window - 1, length, train_stride))
    positives = sum(float(videos[v][1][t]) for v, t in train_items)
    pos_weight = torch.tensor((len(train_items) - positives) / max(positives, 1.0), device=device)

    def batch_windows(items):
        xs = []
        for v, t in items:
            start = max(0, t - window + 1)
            chunk = videos[v][0][start : t + 1]
            if len(chunk) < window:
                chunk = np.concatenate([np.repeat(chunk[:1], window - len(chunk), axis=0), chunk])
            xs.append(chunk)
        return torch.from_numpy(np.stack(xs)).to(device)

    def score_split(ids):
        model.eval()
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
                for k, t in enumerate(ends):  # zero-order hold until the next scored frame
                    nxt = ends[k + 1] if k + 1 < len(ends) else length
                    scores[t:nxt] = values[k]
                scores[: ends[0]] = values[0]
                out[v] = scores
        return out

    best, best_state, bad, history = -1.0, None, 0, []
    rng = random.Random(seed)
    for epoch in range(epochs):
        model.train()
        rng.shuffle(train_items)
        total, steps = 0.0, 0
        for i in range(0, len(train_items), batch_size):
            items = train_items[i : i + batch_size]
            y = torch.tensor([videos[v][1][t] for v, t in items], device=device)
            loss = F.binary_cross_entropy_with_logits(model(batch_windows(items)), y, pos_weight=pos_weight)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total += float(loss)
            steps += 1
        val = frame_auprc(score_split(split["val"]), {v: videos[v][1] for v in split["val"]})
        history.append({"epoch": epoch + 1, "train_loss": total / max(steps, 1), "val_auprc": val})
        if val > best:
            best, bad = val, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(best_state)
    return model, score_split(split["val"]), score_split(split["test"]), history, best


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--method", choices=["causal_tcn", "mistsense"], required=True)
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--window", type=int, default=64)
    args = parser.parse_args()

    set_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    manifest = _read(args.benchmark / "manifest.json")
    split = next(s for s in _read(args.benchmark / "splits.json")["folds"] if int(s["fold"]) == args.fold)
    records = {str(r["video_id"]): r for r in manifest["workers"]}
    needed = sorted(set(split["train"]) | set(split["val"]) | set(split["test"]))
    videos = {v: load_video(records[v]) for v in needed}
    device = torch.device(args.device)
    if args.method == "causal_tcn":
        model, val_scores, test_scores, history, best = run_tcn(videos, split, device, args.seed, args.epochs, args.patience, lr=1e-3)
        config = {"width": 256, "dilations": [1, 2, 4, 8, 16, 32], "dropout": 0.2, "lr": 1e-3}
    else:
        model, val_scores, test_scores, history, best = run_mistsense(
            videos, split, device, args.seed, min(args.epochs, 20), args.patience, lr=3e-4,
            window=args.window, train_stride=8, test_stride=4, batch_size=64,
        )
        config = {"width": 384, "queries": 32, "layers": 2, "window": args.window, "train_stride": 8, "test_stride": 4, "lr": 3e-4}
    out = args.output_root / f"fold_{args.fold}"
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / "validation_scores.npz", **val_scores)
    np.savez_compressed(out / "test_scores.npz", **test_scores)
    torch.save(model.state_dict(), out / "model.pt")
    meta = {
        "method": args.method, "fold": args.fold, "seed": args.seed, "config": config,
        "validation_video_ids": list(split["val"]), "test_video_ids": list(split["test"]),
        "training_video_ids": list(split["train"]), "best_validation_auprc": best,
        "epochs_run": len(history), "history": history,
        "test_annotations_consumed_by_model": False,
        "parameters": sum(p.numel() for p in model.parameters()),
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(f"[done] {args.method} fold={args.fold} seed={args.seed} val_auprc={best:.4f} epochs={len(history)}")


if __name__ == "__main__":
    main()
