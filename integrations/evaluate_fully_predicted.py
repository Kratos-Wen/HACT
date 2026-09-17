#!/usr/bin/env python3
"""Evaluate fully predicted anomaly timelines under one frame protocol.

Every method contributes only segments inferred from cached visual features.
Ground-truth events are read after inference to construct frame targets.  They
are never used to pool recognition outputs or choose predicted boundaries.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


PROJECT = Path(__file__).resolve().parents[1]
WORKSPACE = PROJECT.parent
DEFAULT_BENCHMARK = PROJECT / "outputs" / "impact_benchmark"


def _read(path: Path):
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _average_precision(targets: np.ndarray, scores: np.ndarray) -> float:
    positives = int(targets.sum())
    if not positives:
        return float("nan")
    order = np.argsort(-scores, kind="stable")
    ranked_target = targets[order]
    ranked_score = scores[order]
    true_positive = np.cumsum(ranked_target == 1)
    ends = np.flatnonzero(np.r_[ranked_score[1:] != ranked_score[:-1], True])
    precision = true_positive[ends] / (ends + 1)
    recall = true_positive[ends] / positives
    return float(np.sum(np.diff(np.r_[0.0, recall]) * precision))


def _select_threshold(targets: np.ndarray, scores: np.ndarray) -> float:
    # Zero denotes frames outside every predicted segment.  It is not a valid
    # alarm threshold: selecting it would turn missing temporal support into a
    # positive prediction.  Include the smallest positive float so an
    # all-zero validation timeline still predicts no alarms.
    positive_floor = np.nextafter(0.0, 1.0)
    candidates = np.unique(
        np.r_[positive_floor, 0.5, 1.0, scores[scores > 0.0]]
    )
    selected, best = 0.5, (-1.0, -1.0, -1.0)
    for threshold in candidates:
        prediction = scores >= threshold
        tp = int(np.sum((targets == 1) & prediction))
        fp = int(np.sum((targets == 0) & prediction))
        fn = int(np.sum((targets == 1) & ~prediction))
        denominator = 2 * tp + fp + fn
        f1 = 2 * tp / denominator if denominator else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        key = (f1, recall, -abs(float(threshold) - 0.5))
        if key > best:
            selected, best = float(threshold), key
    return selected


def _metrics(
    targets: np.ndarray,
    scores: np.ndarray,
    decisions: np.ndarray,
    recovery: np.ndarray,
    support: np.ndarray | None = None,
) -> dict:
    tp = int(np.sum((targets == 1) & decisions))
    fp = int(np.sum((targets == 0) & decisions))
    fn = int(np.sum((targets == 1) & ~decisions))
    denominator = 2 * tp + fp + fn
    # A recovery by one hand can overlap an anomaly by the other hand.  Such a
    # frame is a true anomaly on the shared timeline and must not count toward
    # the recovery false-positive denominator.
    recovery_normal = recovery & (targets == 0)
    return {
        "frames": int(len(targets)),
        "anomaly_frames": int(targets.sum()),
        "anomaly_prevalence": float(targets.mean()),
        "anomaly_auprc": _average_precision(targets, scores),
        "anomaly_f1": float(2 * tp / denominator) if denominator else 0.0,
        "precision": float(tp / (tp + fp)) if tp + fp else 0.0,
        "recall": float(tp / (tp + fn)) if tp + fn else 0.0,
        "recovery_frame_fpr": (
            float(decisions[recovery_normal].mean()) if recovery_normal.any() else None
        ),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        **_support_metrics(targets, decisions, recovery_normal, support),
    }


def _support_metrics(
    targets: np.ndarray,
    decisions: np.ndarray,
    recovery_normal: np.ndarray,
    support: np.ndarray | None,
) -> dict:
    """Decoded-support coverage per frame class and the recovery rate on covered frames only.

    Frames outside the predicted support receive score zero and can never be alarmed, so a low
    coverage of recovery frames would lower the recovery false-positive rate without any judgment
    of the recovery.  The covered-only rate removes that route.
    """

    if support is None:
        return {}
    support = np.asarray(support, dtype=bool)
    normal = (targets == 0) & ~recovery_normal
    covered_recovery = recovery_normal & support
    return {
        "support_coverage": float(support.mean()) if len(support) else None,
        "support_coverage_anomaly": float(support[targets == 1].mean()) if (targets == 1).any() else None,
        "support_coverage_normal": float(support[normal].mean()) if normal.any() else None,
        "support_coverage_recovery": float(support[recovery_normal].mean()) if recovery_normal.any() else None,
        "recovery_frame_fpr_covered": (
            float(decisions[covered_recovery].mean()) if covered_recovery.any() else None
        ),
        "recovery_frames": int(recovery_normal.sum()),
        "recovery_frames_covered": int(covered_recovery.sum()),
    }


def _ground_truth(record: dict) -> tuple[np.ndarray, np.ndarray]:
    target = np.zeros(int(record["feature_length"]), dtype=np.int64)
    recovery = np.zeros_like(target, dtype=bool)
    annotation = _read(Path(record["parsed_annotation_path"]))
    for event in annotation.get("events", []):
        start = max(0, int(event["start_frame"]))
        end = min(len(target), int(event["end_frame"]) + 1)
        label = str(event.get("anomaly_label", "normal")).strip().lower()
        if label == "recovery":
            recovery[start:end] = True
        elif label not in {"", "normal", "none", "null"}:
            target[start:end] = 1
    return target, recovery


def _score_video(
    length: int,
    rows: list[dict],
    end_inclusive: bool,
    timeline_mode: str = "event",
) -> np.ndarray:
    if timeline_mode not in {"event", "phase"}:
        raise ValueError(f"Unknown timeline mode: {timeline_mode}")
    score = np.zeros(length, dtype=np.float64)
    support = np.zeros(length, dtype=bool)
    for row in rows:
        if row.get("detected") is False:
            continue
        start = max(0, int(row["start_frame"]))
        end = int(row["end_frame"]) + int(end_inclusive)
        end = min(length, max(start, end))
        if end <= start:
            continue
        support[start:end] = True
        points = row.get("transition_predictions", [])
        if timeline_mode != "phase" or not points:
            score[start:end] = np.maximum(
                score[start:end], float(row["anomaly_score"])
            )
            continue

        # A transition is scored using only observations available at that
        # predicted transition.  Its score is held until the next predicted
        # transition, producing a causal zero-order-hold anomaly timeline.
        # If phases collapse onto one frame, the latest phase is the active one.
        by_frame = {}
        for point in sorted(
            points,
            key=lambda item: (int(item["frame"]), int(item.get("transition", -1))),
        ):
            by_frame[int(point["frame"])] = point
        ordered = [by_frame[frame] for frame in sorted(by_frame)]
        for index, point in enumerate(ordered):
            phase_start = start if index == 0 else max(start, int(point["frame"]))
            phase_end = (
                min(end, int(ordered[index + 1]["frame"]))
                if index + 1 < len(ordered)
                else end
            )
            if phase_end > phase_start:
                score[phase_start:phase_end] = np.maximum(
                    score[phase_start:phase_end], float(point["anomaly_score"])
                )
    _LAST_SUPPORT.append(support)
    return score


# The support masks of the videos scored by the most recent ``_arrays_with_support`` call, in order.
_LAST_SUPPORT: list = []


def _arrays_with_support(
    video_ids: list[str],
    records: dict[str, dict],
    rows: list[dict],
    end_inclusive: bool,
    timeline_mode: str = "event",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Like :func:`_arrays`, with the predicted-support mask as a fourth array."""

    dense = isinstance(rows, dict)
    grouped = {} if dense else _rows_by_video(rows)
    targets, scores, recoveries, supports = [], [], [], []
    for video_id in video_ids:
        record = records[video_id]
        target, recovery = _ground_truth(record)
        if dense:
            score = np.zeros(len(target), dtype=np.float64)
            values = np.asarray(rows.get(video_id, []), dtype=np.float64)
            length = min(len(values), len(target))
            score[:length] = values[:length]
            support = np.zeros(len(target), dtype=bool)
            support[:length] = True
        else:
            _LAST_SUPPORT.clear()
            score = _score_video(
                len(target), grouped.get(video_id, []), end_inclusive, timeline_mode
            )
            support = _LAST_SUPPORT.pop()
        targets.append(target)
        scores.append(score)
        recoveries.append(recovery)
        supports.append(support)
    return (
        np.concatenate(targets),
        np.concatenate(scores),
        np.concatenate(recoveries),
        np.concatenate(supports),
    )


def _state_decision_video(
    length: int,
    rows: list[dict],
    end_inclusive: bool,
    timeline_mode: str = "event",
) -> np.ndarray:
    """Return the causal MAP error-state decision on one video timeline.

    This control has no scalar alarm threshold: a covered frame is anomalous
    exactly when ``error`` is the most probable latent state.  Overlapping
    hand streams are combined by logical OR, matching the max-score fusion
    used by :func:`_score_video`.
    """

    if timeline_mode not in {"event", "phase"}:
        raise ValueError(f"Unknown timeline mode: {timeline_mode}")
    decision = np.zeros(length, dtype=bool)

    def is_error(item: dict) -> bool:
        probabilities = np.asarray(item["state_probabilities"], dtype=np.float64)
        if probabilities.ndim != 1 or len(probabilities) < 2:
            raise ValueError("State-MAP decisions require at least two states")
        return int(np.argmax(probabilities)) == 1

    for row in rows:
        if row.get("detected") is False:
            continue
        start = max(0, int(row["start_frame"]))
        end = int(row["end_frame"]) + int(end_inclusive)
        end = min(length, max(start, end))
        if end <= start:
            continue
        points = row.get("transition_predictions", [])
        if timeline_mode != "phase" or not points:
            if is_error(row):
                decision[start:end] = True
            continue

        by_frame = {}
        for point in sorted(
            points,
            key=lambda item: (int(item["frame"]), int(item.get("transition", -1))),
        ):
            by_frame[int(point["frame"])] = point
        ordered = [by_frame[frame] for frame in sorted(by_frame)]
        for index, point in enumerate(ordered):
            phase_start = start if index == 0 else max(start, int(point["frame"]))
            phase_end = (
                min(end, int(ordered[index + 1]["frame"]))
                if index + 1 < len(ordered)
                else end
            )
            if phase_end > phase_start and is_error(point):
                decision[phase_start:phase_end] = True
    return decision


def _rows_by_video(rows: list[dict]) -> dict[str, list[dict]]:
    grouped: dict[str, list[dict]] = {}
    for row in rows:
        grouped.setdefault(str(row["video_id"]), []).append(row)
    return grouped


def _latent_state_count(rows: list[dict]) -> int:
    counts = set()
    for row in rows:
        if "state_probabilities" in row:
            counts.add(len(row["state_probabilities"]))
        for point in row.get("transition_predictions", []):
            if "state_probabilities" in point:
                counts.add(len(point["state_probabilities"]))
    if len(counts) != 1:
        raise ValueError(
            "Uniform-state decisions require one consistent latent-state dimension"
        )
    count = counts.pop()
    if count < 2:
        raise ValueError("At least two latent states are required")
    return count


def _arrays(
    video_ids: list[str],
    records: dict[str, dict],
    rows: list[dict],
    end_inclusive: bool,
    timeline_mode: str = "event",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    dense = isinstance(rows, dict)
    grouped = {} if dense else _rows_by_video(rows)
    targets, scores, recoveries = [], [], []
    for video_id in video_ids:
        record = records[video_id]
        target, recovery = _ground_truth(record)
        if dense:
            # Dense per-frame scores from a frame-level model; missing frames
            # at the end (feature/annotation length mismatch) receive zero.
            score = np.zeros(len(target), dtype=np.float64)
            values = np.asarray(rows.get(video_id, []), dtype=np.float64)
            length = min(len(values), len(target))
            score[:length] = values[:length]
        else:
            score = _score_video(
                len(target), grouped.get(video_id, []), end_inclusive, timeline_mode
            )
        targets.append(target)
        scores.append(score)
        recoveries.append(recovery)
    return np.concatenate(targets), np.concatenate(scores), np.concatenate(recoveries)


def _state_decisions(
    video_ids: list[str],
    records: dict[str, dict],
    rows: list[dict],
    end_inclusive: bool,
    timeline_mode: str = "event",
) -> np.ndarray:
    grouped = _rows_by_video(rows)
    return np.concatenate(
        [
            _state_decision_video(
                int(records[video_id]["feature_length"]),
                grouped.get(video_id, []),
                end_inclusive,
                timeline_mode,
            )
            for video_id in video_ids
        ]
    )


def _prediction_candidates(method: str, root: Path, fold: int, seed: int = 0) -> list[dict]:
    if method == "hact":
        directory = root / f"seed_{seed}" / f"fold_{fold}"
        manifest_path = directory / "rescore_manifest.json"
        manifest = _read(manifest_path) if manifest_path.exists() else {}
        return [
            {
                "name": f"seed_{seed}",
                "validation": _read(directory / "validation_event_predictions.json"),
                "test": _read(directory / "predicted_event_predictions.json"),
                "end_inclusive": True,
                "source": str(directory),
                "validation_video_ids": manifest.get("validation_videos"),
                "test_video_ids": manifest.get("test_videos"),
            }
        ]
    if method == "egoper":
        candidates = []
        for path in sorted((root / f"fold_{fold}").glob("epoch_*.json")):
            payload = _read(path)
            candidates.append(
                {
                    "name": path.stem,
                    "validation": payload["validation_rows"],
                    "test": payload["test_rows"],
                    "end_inclusive": False,
                    "source": str(path),
                    "validation_video_ids": payload.get("validation_video_ids"),
                    "test_video_ids": payload.get("test_video_ids"),
                }
            )
        return candidates
    if method == "dense":
        directory = root / f"fold_{fold}"
        meta = _read(directory / "meta.json")
        with np.load(directory / "validation_scores.npz") as payload:
            validation = {key: np.asarray(payload[key], dtype=np.float64) for key in payload.files}
        with np.load(directory / "test_scores.npz") as payload:
            test = {key: np.asarray(payload[key], dtype=np.float64) for key in payload.files}
        return [
            {
                "name": str(meta.get("method", "dense")),
                "validation": validation,
                "test": test,
                "end_inclusive": True,
                "source": str(directory),
                "validation_video_ids": meta.get("validation_video_ids"),
                "test_video_ids": meta.get("test_video_ids"),
            }
        ]
    if method == "prego":
        path = root / f"fold_{fold}" / "prego_predicted_results.json"
        payload = _read(path)
        events_path = path.with_name("prego_predicted_events.json")
        events_payload = _read(events_path) if events_path.exists() else {}
        return [
            {
                "name": "native_binary",
                "validation": None,
                "test": payload["rows"],
                "end_inclusive": False,
                "source": str(path),
                "validation_video_ids": payload.get("validation_video_ids"),
                "test_video_ids": payload.get("test_video_ids"),
                "protocol": payload.get("protocol"),
                "inference": payload.get("inference"),
                "recognition_provenance": events_payload.get(
                    "recognition_provenance"
                ),
            }
        ]
    raise ValueError(method)


def _validate_declared_video_ids(
    method: str, fold: int, split: dict, selected: dict
) -> None:
    for partition, key in (
        ("val", "validation_video_ids"),
        ("test", "test_video_ids"),
    ):
        declared = selected.get(key)
        if declared is not None and list(declared) != list(split[partition]):
            raise ValueError(
                f"{method} fold {fold} declares the wrong {partition} videos: "
                f"{declared} != {split[partition]}"
            )


def _validate_prediction_provenance(method: str, fold: int, selected: dict) -> None:
    forbidden = {
        "target",
        "target_name",
        "recovery",
        "matched_event_id",
        "match_iou",
        "anomaly_label",
        "has_anomaly",
        "labels_error",
        "error_description",
    }
    for partition in ("validation", "test"):
        rows = selected.get(partition) or []
        if isinstance(rows, dict):
            continue
        leaked = sorted(set().union(*(set(row) & forbidden for row in rows)))
        if leaked:
            raise ValueError(
                f"{method} fold {fold} {partition} predictions contain "
                f"annotation fields: {leaked}"
            )
    if method != "prego":
        return
    if selected.get("protocol") != "fully_predicted_miniroad_segments":
        raise ValueError(f"PREGO fold {fold} is not a fully predicted run")
    recognition = selected.get("recognition_provenance")
    if not isinstance(recognition, dict):
        raise ValueError(f"PREGO fold {fold} lacks recognition provenance")
    if not isinstance(recognition.get("seed"), int):
        raise ValueError(f"PREGO fold {fold} recognition does not record its seed")
    if recognition.get("test_annotations_consumed_by_model") is not False:
        raise ValueError(f"PREGO fold {fold} recognition consumed test annotations")
    inference = selected.get("inference")
    if not isinstance(inference, dict) or not isinstance(inference.get("seed"), int):
        raise ValueError(f"PREGO fold {fold} language inference does not record its seed")


def evaluate(
    method: str,
    root: Path,
    benchmark: Path,
    timeline_mode: str = "event",
    decision_mode: str = "validation_f1",
    seed: int = 0,
) -> dict:
    if decision_mode not in {
        "validation_f1",
        "fixed_half",
        "uniform_state",
        "state_argmax",
    }:
        raise ValueError(f"Unknown decision mode: {decision_mode}")
    manifest = _read(benchmark / "manifest.json")
    splits = _read(benchmark / "splits.json")["folds"]
    records = {str(row["video_id"]): row for row in manifest["workers"]}
    pooled_target, pooled_score, pooled_decision, pooled_recovery = [], [], [], []
    pooled_support = []
    fold_reports = []
    for split in splits:
        fold = int(split["fold"])
        candidates = _prediction_candidates(method, root, fold, seed=seed)
        if not candidates:
            raise FileNotFoundError(f"No {method} candidates found for fold {fold}")
        ranked = []
        for candidate in candidates:
            if candidate["validation"] is None:
                ranked.append((float("nan"), candidate))
                continue
            target, score, _ = _arrays(
                split["val"],
                records,
                candidate["validation"],
                bool(candidate["end_inclusive"]),
                timeline_mode,
            )
            ranked.append((_average_precision(target, score), candidate))
        selected = (
            ranked[0][1]
            if ranked[0][1]["validation"] is None
            else max(ranked, key=lambda item: item[0])[1]
        )
        _validate_declared_video_ids(method, fold, split, selected)
        _validate_prediction_provenance(method, fold, selected)
        if selected["validation"] is None:
            threshold = 0.5
            validation_auprc = None
        else:
            val_target, val_score, _ = _arrays(
                split["val"],
                records,
                selected["validation"],
                bool(selected["end_inclusive"]),
                timeline_mode,
            )
            if decision_mode == "uniform_state":
                threshold = 1.0 / _latent_state_count(selected["validation"])
            elif decision_mode == "fixed_half":
                threshold = 0.5
            elif decision_mode == "state_argmax":
                _latent_state_count(selected["validation"])
                threshold = None
            else:
                threshold = _select_threshold(val_target, val_score)
            validation_auprc = _average_precision(val_target, val_score)
        target, score, recovery, support = _arrays_with_support(
            split["test"],
            records,
            selected["test"],
            bool(selected["end_inclusive"]),
            timeline_mode,
        )
        decision = (
            _state_decisions(
                split["test"],
                records,
                selected["test"],
                bool(selected["end_inclusive"]),
                timeline_mode,
            )
            if decision_mode == "state_argmax"
            else score >= float(threshold)
        )
        pooled_target.append(target)
        pooled_score.append(score)
        pooled_decision.append(decision)
        pooled_recovery.append(recovery)
        pooled_support.append(support)
        fold_reports.append(
            {
                "fold": fold,
                "selected": selected["name"],
                "source": selected["source"],
                "end_inclusive": bool(selected["end_inclusive"]),
                "validation_auprc": validation_auprc,
                "threshold": threshold,
                "test": _metrics(target, score, decision, recovery, support),
            }
        )
    target = np.concatenate(pooled_target)
    score = np.concatenate(pooled_score)
    decision = np.concatenate(pooled_decision)
    recovery = np.concatenate(pooled_recovery)
    support = np.concatenate(pooled_support)
    return {
        "method": method,
        "protocol": "cached_visual_features_to_predicted_anomaly_timeline",
        "timeline_mode": timeline_mode,
        "decision_mode": "native_binary" if method == "prego" else decision_mode,
        "test_oracle_inputs": [],
        "folds": fold_reports,
        "pooled": _metrics(target, score, decision, recovery, support),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("method", choices=("hact", "egoper", "prego", "dense"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--benchmark", type=Path, default=DEFAULT_BENCHMARK)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--timeline",
        choices=("event", "phase"),
        default="event",
        help="Use one event score or causal start/onset/end transition scores",
    )
    parser.add_argument(
        "--decision",
        choices=("validation_f1", "fixed_half", "uniform_state", "state_argmax"),
        default="validation_f1",
        help=(
            "Select validation F1, use a fixed binary half-probability, the "
            "reciprocal latent-state count, or the threshold-free MAP state"
        ),
    )
    parser.add_argument("--seed", type=int, default=0,
                        help="which seed subdirectory of the prediction root to evaluate")
    args = parser.parse_args()
    report = evaluate(
        args.method,
        args.root.resolve(),
        args.benchmark.resolve(),
        timeline_mode=args.timeline,
        decision_mode=args.decision,
        seed=args.seed,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report["pooled"], indent=2))


if __name__ == "__main__":
    main()
