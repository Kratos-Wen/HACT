#!/usr/bin/env python3
"""Recovery false-positive rate at a validation-selected operating point.

For every fold the threshold is the largest score at which the pooled recall on the fold's validation
participants reaches the requested level; the test recordings are then read once at that threshold.
The pooled test decisions give the recovery false-positive rate on all and on covered recovery frames,
the achieved test recall, and the false-positive rate on normal frames.  No test label enters the
threshold.  Timelines and candidate selection are those of integrations/evaluate_fully_predicted.py.
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate_fully_predicted import _arrays, _arrays_with_support, _average_precision, _metrics, _prediction_candidates, _read  # noqa: E402


def threshold_at_recall(target: np.ndarray, score: np.ndarray, level: float):
    """Largest threshold whose recall on (target, score) reaches ``level``; None if unreachable."""
    positives = int(target.sum())
    if positives == 0: return None
    order = np.argsort(-score, kind="stable"); ranked_t = target[order]; ranked_s = score[order]
    cumulative = np.cumsum(ranked_t == 1) / positives
    reached = np.flatnonzero(cumulative >= level)
    if not len(reached) or ranked_s[reached[0]] <= 0.0: return None
    return float(ranked_s[reached[0]])


def evaluate(method: str, root: Path, benchmark: Path, timeline: str, seed: int, levels):
    manifest = _read(benchmark / "manifest.json"); splits = _read(benchmark / "splits.json")["folds"]
    records = {str(r["video_id"]): r for r in manifest["workers"]}
    per_level = {lvl: dict(targets=[], scores=[], recoveries=[], decisions=[], supports=[], thresholds=[]) for lvl in levels}
    for split in splits:
        fold = int(split["fold"]); candidates = _prediction_candidates(method, root, fold, seed=seed)
        ranked = []
        for c in candidates:
            if c["validation"] is None: ranked.append((float("nan"), c)); continue
            t, s, _ = _arrays(split["val"], records, c["validation"], bool(c["end_inclusive"]), timeline)
            ranked.append((_average_precision(t, s), c))
        selected = ranked[0][1] if ranked[0][1]["validation"] is None else max(ranked, key=lambda x: x[0])[1]
        if selected["validation"] is None: raise SystemExit(f"{method}: no validation predictions, no operating point can be selected")
        vt, vs, _ = _arrays(split["val"], records, selected["validation"], bool(selected["end_inclusive"]), timeline)
        target, score, recovery, support = _arrays_with_support(split["test"], records, selected["test"], bool(selected["end_inclusive"]), timeline)
        for lvl in levels:
            thr = threshold_at_recall(vt, vs, lvl)
            d = per_level[lvl]; d["targets"].append(target); d["scores"].append(score); d["recoveries"].append(recovery); d["supports"].append(support)
            d["thresholds"].append(thr); d["decisions"].append(score >= thr if thr is not None else np.zeros(len(score), bool))
    out = []
    for lvl in levels:
        d = per_level[lvl]
        target = np.concatenate(d["targets"]); score = np.concatenate(d["scores"]); recovery = np.concatenate(d["recoveries"])
        decision = np.concatenate(d["decisions"]); support = np.concatenate(d["supports"])
        m = _metrics(target, score, decision, recovery, support)
        normal = (target == 0) & ~recovery
        out.append({"recall_level": lvl, "fold_thresholds": d["thresholds"], "recall": m["recall"], "precision": m["precision"],
                    "normal_frame_fpr": float(decision[normal].mean()) if normal.any() else None,
                    "recovery_frame_fpr": m["recovery_frame_fpr"], "recovery_frame_fpr_covered": m.get("recovery_frame_fpr_covered"),
                    "f1": m["anomaly_f1"], "auprc": m["anomaly_auprc"]})
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("method", choices=["hact", "egoper", "dense"]); ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--benchmark", type=Path, required=True); ap.add_argument("--timeline", choices=["event", "phase"], default="event")
    ap.add_argument("--recall-levels", type=float, nargs="+", default=[0.3, 0.4, 0.5]); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--output", type=Path, required=True)
    a = ap.parse_args()
    rows = evaluate(a.method, a.root.resolve(), a.benchmark.resolve(), a.timeline, a.seed, a.recall_levels)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps({"method": a.method, "timeline_mode": a.timeline, "seed": a.seed, "validation_recall": rows}, indent=2) + "\n")
    for r in rows: print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items() if k != "fold_thresholds"})


if __name__ == "__main__":
    main()
