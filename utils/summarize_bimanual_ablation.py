"""Pool out-of-fold predictions and summarize bimanual-transition ablations."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections import defaultdict
from typing import Dict, List, Mapping, Sequence, Tuple

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_PIPELINE = os.path.join(_ROOT, "pipeline")
if _PIPELINE not in sys.path:
    sys.path.insert(0, _PIPELINE)

from bimanual_transition_data import ANOMALY_TYPES  # noqa: E402
from bimanual_transition_metrics import evaluate_event_predictions  # noqa: E402
from procedure_scoring import evaluate_procedure_scores  # noqa: E402


METRICS = (
    "anomaly_auprc",
    "anomaly_f1",
    "type_macro_f1_supported",
    "temporal_auprc",
    "recovery_fpr",
    "brier",
    "ece",
    "event_detection_f1",
    "procedure_score_mae",
    "procedure_score_spearman",
    "procedure_score_pearson",
)

PREDICTION_FILES = {
    "predicted_event": "predicted_event_predictions.json",
    "oracle_event": "oracle_event_predictions.json",
}


def _json_safe(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def discover(root: str) -> List[dict]:
    rows = []
    for directory, _, files in os.walk(root):
        if "metrics.json" not in files or "_shared_observation" in directory:
            continue
        path = os.path.join(directory, "metrics.json")
        with open(path, encoding="utf-8") as f:
            payload = json.load(f)
        # Analysis utilities also write per-fold ``metrics.json`` files for
        # seed ensembles.  Those are aggregate artifacts rather than training
        # runs and intentionally do not carry run metadata.
        if not all(key in payload for key in ("variant", "seed", "fold")):
            continue
        payload["path"] = path
        config_path = os.path.join(directory, "effective_config.json")
        if os.path.exists(config_path):
            with open(config_path, encoding="utf-8") as f:
                effective_config = json.load(f)
            payload["expected_folds"] = int(
                effective_config.get("split", {}).get("folds", 0)
            )
        rows.append(payload)
    return rows


def _finite_summary(values: Sequence[float]) -> dict:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    return {
        "mean": float(np.mean(finite)) if finite else None,
        "std": (
            float(np.std(finite, ddof=1))
            if len(finite) > 1
            else 0.0 if finite else None
        ),
        "n": len(finite),
    }


def _load_fold_predictions(row: Mapping[str, object], evaluation: str) -> List[dict]:
    directory = os.path.dirname(str(row["path"]))
    path = os.path.join(directory, PREDICTION_FILES[evaluation])
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Missing prediction file for pooled CV metrics: {path}"
        )
    with open(path, encoding="utf-8") as f:
        events = json.load(f)
    threshold = row.get("threshold")
    for event in events:
        if "decision_threshold" not in event:
            if threshold is None:
                raise ValueError(f"No validation threshold stored for {path}")
            event["decision_threshold"] = float(threshold)
    return events


def pool_seed_results(rows: Sequence[dict], evaluation: str) -> List[dict]:
    grouped: Dict[Tuple[str, int], List[dict]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["variant"]), int(row["seed"]))].append(row)

    results = []
    for (variant, seed), members in sorted(grouped.items()):
        folds = [int(member["fold"]) for member in members]
        if len(folds) != len(set(folds)):
            raise ValueError(f"Duplicate fold for {variant}, seed={seed}: {folds}")
        events = []
        procedure_rows = []
        seen_videos = set()
        for member in sorted(members, key=lambda value: int(value["fold"])):
            fold_events = _load_fold_predictions(member, evaluation)
            fold_videos = {str(event["video_id"]) for event in fold_events}
            overlap = seen_videos & fold_videos
            if overlap:
                raise ValueError(
                    f"Videos occur in multiple test folds for {variant}, seed={seed}: "
                    f"{sorted(overlap)}"
                )
            seen_videos.update(fold_videos)
            events.extend(fold_events)
            if evaluation == "predicted_event":
                procedure_path = os.path.join(
                    os.path.dirname(str(member["path"])), "procedure_scores.json"
                )
                if os.path.exists(procedure_path):
                    with open(procedure_path, encoding="utf-8") as f:
                        procedure_rows.extend(json.load(f))
        expected_values = [int(member.get("expected_folds", 0)) for member in members]
        expected_folds = max(expected_values) if expected_values else 0
        metrics = _json_safe(
            evaluate_event_predictions(events, ANOMALY_TYPES, threshold=None)
        )
        procedure_metrics = evaluate_procedure_scores(procedure_rows)
        for metric in ("mae", "spearman", "pearson"):
            if metric in procedure_metrics:
                metrics[f"procedure_score_{metric}"] = _json_safe(
                    procedure_metrics[metric]
                )
        results.append(
            {
                "variant": variant,
                "seed": seed,
                "folds": sorted(folds),
                "expected_folds": expected_folds,
                "complete": bool(expected_folds and len(folds) == expected_folds),
                "metrics": metrics,
            }
        )
    return results


def aggregate(rows: List[dict], evaluation: str) -> dict:
    seed_results = pool_seed_results(rows, evaluation)
    grouped: Dict[str, List[dict]] = defaultdict(list)
    for result in seed_results:
        grouped[result["variant"]].append(result)

    variants = {}
    for variant, members in sorted(grouped.items()):
        summary = {
            "seeds": len(members),
            "fold_runs": sum(len(member["folds"]) for member in members),
            "complete_seeds": sum(bool(member["complete"]) for member in members),
        }
        for metric in METRICS:
            summary[metric] = _finite_summary(
                [
                    (
                        member["metrics"].get(metric, float("nan"))
                        if member["metrics"].get(metric) is not None
                        else float("nan")
                    )
                    for member in members
                ]
            )
        variants[variant] = summary

    paired = {}
    full_by_seed = {
        int(result["seed"]): result
        for result in seed_results
        if result["variant"] == "full"
    }
    for variant, members in sorted(grouped.items()):
        if variant == "full":
            continue
        matched = []
        for member in members:
            full = full_by_seed.get(int(member["seed"]))
            if (
                full is not None
                and member["complete"]
                and full["complete"]
                and member["folds"] == full["folds"]
            ):
                matched.append((member, full))
        comparison = {
            "paired_seeds": len(matched),
            "delta_definition": f"{variant} - full",
        }
        for metric in METRICS:
            deltas = []
            for member, full in matched:
                value = member["metrics"].get(metric)
                reference = full["metrics"].get(metric)
                if value is not None and reference is not None:
                    deltas.append(float(value) - float(reference))
            comparison[metric] = _finite_summary(deltas)
        paired[variant] = comparison

    return {
        "evaluation": evaluation,
        "variants": variants,
        "paired_vs_full": paired,
        "seed_results": seed_results,
    }


def _cell(summary: Mapping[str, object], metric: str) -> str:
    row = summary[metric]
    if row["mean"] is None:
        return "-"
    return f"{row['mean']:.3f} +/- {row['std']:.3f}"


def markdown(report: dict) -> str:
    lines = [
        "| variant | seeds | folds | complete | anomaly AUPRC | anomaly F1 | "
        "type macro-F1 | temporal AUPRC | recovery FPR | score MAE | score rho |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for variant, summary in report["variants"].items():
        cells = [
            variant,
            str(summary["seeds"]),
            str(summary["fold_runs"]),
            str(summary["complete_seeds"]),
            _cell(summary, "anomaly_auprc"),
            _cell(summary, "anomaly_f1"),
            _cell(summary, "type_macro_f1_supported"),
            _cell(summary, "temporal_auprc"),
            _cell(summary, "recovery_fpr"),
            _cell(summary, "procedure_score_mae"),
            _cell(summary, "procedure_score_spearman"),
        ]
        lines.append("| " + " | ".join(cells) + " |")

    if report["paired_vs_full"]:
        lines.extend(
            [
                "",
                "Paired deltas use the same seed and completed fold set and are "
                "reported as ablation minus `full`. For error metrics, a negative "
                "delta is an improvement.",
                "",
                "| variant - full | paired seeds | anomaly AUPRC | anomaly F1 | "
                "temporal AUPRC | recovery FPR |",
                "|---|---:|---:|---:|---:|---:|",
            ]
        )
        for variant, summary in report["paired_vs_full"].items():
            cells = [
                variant,
                str(summary["paired_seeds"]),
                _cell(summary, "anomaly_auprc"),
                _cell(summary, "anomaly_f1"),
                _cell(summary, "temporal_auprc"),
                _cell(summary, "recovery_fpr"),
            ]
            lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="outputs/bimanual_transition_ablation")
    parser.add_argument(
        "--evaluation",
        choices=("predicted_event", "oracle_event"),
        default="predicted_event",
    )
    parser.add_argument("--out_json", default=None)
    parser.add_argument("--out_md", default=None)
    args = parser.parse_args()
    rows = discover(args.root)
    if not rows:
        raise SystemExit(f"No metrics.json files found under {args.root}")
    report = aggregate(rows, args.evaluation)
    table = markdown(report)
    print(table)
    if args.out_json:
        os.makedirs(os.path.dirname(os.path.abspath(args.out_json)), exist_ok=True)
        with open(args.out_json, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
    if args.out_md:
        os.makedirs(os.path.dirname(os.path.abspath(args.out_md)), exist_ok=True)
        with open(args.out_md, "w", encoding="utf-8") as f:
            f.write(table + "\n")


if __name__ == "__main__":
    main()
