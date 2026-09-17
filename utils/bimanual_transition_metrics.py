"""Dependency-light metrics for event-level procedural anomaly assessment."""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np


def softmax(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    values = np.asarray(logits, dtype=np.float64) / max(float(temperature), 1e-6)
    values = values - values.max(axis=-1, keepdims=True)
    exp = np.exp(values)
    return exp / exp.sum(axis=-1, keepdims=True).clip(min=1e-12)


def hierarchical_probabilities(
    binary_logits: np.ndarray,
    type_logits: np.ndarray,
    binary_temperature: float = 1.0,
    type_temperature: float = 1.0,
    supported_types: Optional[Sequence[bool]] = None,
) -> np.ndarray:
    """Compose p(normal) and p(type | anomaly) without unsupported classes."""

    binary = softmax(np.asarray(binary_logits), binary_temperature)
    if binary.shape[-1] not in {2, 3}:
        raise ValueError("Execution-state logits must contain two or three classes")
    conditional_logits = np.asarray(type_logits, dtype=np.float64).copy()
    if supported_types is not None:
        support = np.asarray(supported_types, dtype=bool)
        if support.shape != conditional_logits.shape[-1:]:
            raise ValueError("supported_types does not match type_logits")
        if not np.any(support):
            raise ValueError("At least one anomaly subtype must be supported")
        conditional_logits[..., ~support] = -1e30
    conditional = softmax(conditional_logits, type_temperature)
    error = binary[..., 1:2]
    non_error = 1.0 - error
    return np.concatenate((non_error, error * conditional), axis=-1)


def hierarchical_decision(probabilities: Sequence[float], threshold: float) -> int:
    """Apply the binary decision first, then choose a conditional subtype."""

    values = np.asarray(probabilities, dtype=np.float64)
    if values.ndim != 1 or len(values) < 2:
        raise ValueError("Expected one normal probability and anomaly subtypes")
    if float(1.0 - values[0]) < float(threshold):
        return 0
    return int(1 + np.argmax(values[1:]))


def _temperatures(value: object) -> Tuple[float, float]:
    if isinstance(value, Mapping):
        return float(value.get("binary", 1.0)), float(value.get("type", 1.0))
    scalar = float(value)
    return scalar, scalar


def average_precision(targets: Sequence[int], scores: Sequence[float]) -> float:
    y = np.asarray(targets, dtype=np.int64)
    s = np.asarray(scores, dtype=np.float64)
    positives = int((y == 1).sum())
    if positives == 0:
        return float("nan")
    order = np.argsort(-s, kind="stable")
    ranked = y[order]
    ranked_scores = s[order]
    true_positives = np.cumsum(ranked == 1)
    # Evaluate once at the end of each tied-score group. This matches the
    # threshold-based AP definition and makes the result order-invariant.
    group_ends = np.flatnonzero(np.r_[ranked_scores[1:] != ranked_scores[:-1], True])
    precision = true_positives[group_ends] / (group_ends + 1)
    recall = true_positives[group_ends] / positives
    return float(np.sum(np.diff(np.r_[0.0, recall]) * precision))


def binary_counts(
    targets: np.ndarray, predictions: np.ndarray
) -> Tuple[int, int, int, int]:
    tp = int(np.sum((targets == 1) & (predictions == 1)))
    fp = int(np.sum((targets == 0) & (predictions == 1)))
    fn = int(np.sum((targets == 1) & (predictions == 0)))
    tn = int(np.sum((targets == 0) & (predictions == 0)))
    return tp, fp, fn, tn


def f1_from_counts(tp: int, fp: int, fn: int) -> float:
    denominator = 2 * tp + fp + fn
    return float(2 * tp / denominator) if denominator else 0.0


def select_f1_threshold(targets: Sequence[int], scores: Sequence[float]) -> float:
    y = np.asarray(targets, dtype=np.int64)
    s = np.asarray(scores, dtype=np.float64)
    if len(y) == 0 or not np.any(y == 1):
        return 0.5
    unique_scores = np.unique(s)
    midpoints = (unique_scores[:-1] + unique_scores[1:]) / 2.0
    candidates = np.unique(
        np.concatenate(([1e-6, 0.5, 1.0], unique_scores[unique_scores > 0], midpoints))
    )
    best = (-1.0, -1.0, 0.5)
    for threshold in candidates:
        predictions = (s >= threshold).astype(np.int64)
        tp, fp, fn, _ = binary_counts(y, predictions)
        f1 = f1_from_counts(tp, fp, fn)
        # Recall breaks F1 ties, then proximity to 0.5 gives stable choices.
        recall = tp / (tp + fn) if tp + fn else 0.0
        key = (f1, recall, -abs(float(threshold) - 0.5))
        if key > best:
            best = key
            selected = float(threshold)
    return selected


def expected_calibration_error(
    targets: Sequence[int], scores: Sequence[float], bins: int = 10
) -> float:
    y = np.asarray(targets, dtype=np.float64)
    s = np.asarray(scores, dtype=np.float64)
    if not len(y):
        return float("nan")
    boundaries = np.linspace(0.0, 1.0, bins + 1)
    total = 0.0
    for index in range(bins):
        if index == bins - 1:
            mask = (s >= boundaries[index]) & (s <= boundaries[index + 1])
        else:
            mask = (s >= boundaries[index]) & (s < boundaries[index + 1])
        if np.any(mask):
            total += float(mask.mean()) * abs(
                float(s[mask].mean()) - float(y[mask].mean())
            )
    return float(total)


def aggregate_transition_predictions(
    rows: Iterable[Mapping[str, object]],
    anomaly_types: Sequence[str],
    temperature: object = 1.0,
) -> List[dict]:
    """Aggregate phase views while retaining their causal timeline scores."""

    grouped: Dict[Tuple[str, str], List[Mapping[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["video_id"]), str(row["event_id"]))].append(row)

    events = []
    for (video_id, event_id), members in sorted(grouped.items()):
        labels = {int(member["target"]) for member in members}
        recoveries = {bool(member.get("recovery", False)) for member in members}
        hands = {int(member.get("hand", -1)) for member in members}
        factors = {
            name: {int(member.get(name, -1)) for member in members}
            for name in ("verb", "part", "tool")
        }
        if len(labels) != 1 or len(recoveries) != 1 or len(hands) != 1:
            raise ValueError(
                f"Inconsistent transition targets for {video_id}/{event_id}"
            )
        if any(len(values) != 1 for values in factors.values()):
            raise ValueError(
                f"Inconsistent transition factors for {video_id}/{event_id}"
            )
        mean_logits = np.mean(
            [np.asarray(member["logits"], dtype=np.float64) for member in members],
            axis=0,
        )
        binary_logits = None
        type_logits = None
        supported_types = None
        if all(
            "binary_logits" in member and "type_logits" in member for member in members
        ):
            binary_logits = np.mean(
                [
                    np.asarray(member["binary_logits"], dtype=np.float64)
                    for member in members
                ],
                axis=0,
            )
            type_logits = np.mean(
                [
                    np.asarray(member["type_logits"], dtype=np.float64)
                    for member in members
                ],
                axis=0,
            )
            support_values = {
                tuple(bool(value) for value in member["supported_anomaly_types"])
                for member in members
            }
            if len(support_values) != 1:
                raise ValueError(
                    f"Inconsistent anomaly support for {video_id}/{event_id}"
                )
            supported_types = list(support_values.pop())
            binary_temperature, type_temperature = _temperatures(temperature)
            probabilities = hierarchical_probabilities(
                binary_logits,
                type_logits,
                binary_temperature,
                type_temperature,
                supported_types,
            )
        else:
            probabilities = softmax(mean_logits, temperature=float(temperature))
        evidence_keys = (
            set.intersection(
                *[set(member.get("evidence", {}).keys()) for member in members]
            )
            if members
            else set()
        )
        evidence = {
            key: float(np.mean([float(member["evidence"][key]) for member in members]))
            for key in sorted(evidence_keys)
        }
        target = labels.pop()
        predicted_type = hierarchical_decision(probabilities, threshold=0.5)
        event = {
            "video_id": video_id,
            "event_id": event_id,
            "target": target,
            "target_name": anomaly_types[target],
            "recovery": recoveries.pop(),
            "hand": hands.pop(),
            "verb": factors["verb"].pop(),
            "part": factors["part"].pop(),
            "tool": factors["tool"].pop(),
            "start_frame": int(
                min(
                    int(member.get("event_start_frame", member["frame"]))
                    for member in members
                )
            ),
            "end_frame": int(
                max(
                    int(member.get("event_end_frame", member["frame"]))
                    for member in members
                )
            ),
            "logits": mean_logits.tolist(),
            "probabilities": probabilities.tolist(),
            "anomaly_score": float(1.0 - probabilities[0]),
            "predicted_type": predicted_type,
            "predicted_type_name": anomaly_types[predicted_type],
            "evidence": evidence,
        }
        if binary_logits is not None and type_logits is not None:
            event.update(
                {
                    "binary_logits": binary_logits.tolist(),
                    "type_logits": type_logits.tolist(),
                    "supported_anomaly_types": supported_types,
                }
            )
            if len(binary_logits) == 3:
                event["state_probabilities"] = softmax(
                    binary_logits, binary_temperature
                ).tolist()
        transition_predictions = []
        for member in sorted(
            members,
            key=lambda item: (int(item["frame"]), int(item.get("transition", -1))),
        ):
            member_logits = np.asarray(member["logits"], dtype=np.float64)
            point = {
                "frame": int(member["frame"]),
                "transition": int(member.get("transition", -1)),
                "logits": member_logits.tolist(),
            }
            if "binary_logits" in member and "type_logits" in member:
                member_binary = np.asarray(
                    member["binary_logits"], dtype=np.float64
                )
                member_type = np.asarray(member["type_logits"], dtype=np.float64)
                member_support = list(
                    bool(value) for value in member["supported_anomaly_types"]
                )
                binary_temperature, type_temperature = _temperatures(temperature)
                member_probabilities = hierarchical_probabilities(
                    member_binary,
                    member_type,
                    binary_temperature,
                    type_temperature,
                    member_support,
                )
                point.update(
                    {
                        "binary_logits": member_binary.tolist(),
                        "type_logits": member_type.tolist(),
                        "supported_anomaly_types": member_support,
                    }
                )
                if len(member_binary) == 3:
                    point["state_probabilities"] = softmax(
                        member_binary, binary_temperature
                    ).tolist()
            else:
                member_probabilities = softmax(
                    member_logits, temperature=float(temperature)
                )
            member_type_index = hierarchical_decision(
                member_probabilities, threshold=0.5
            )
            point.update(
                {
                    "probabilities": member_probabilities.tolist(),
                    "anomaly_score": float(1.0 - member_probabilities[0]),
                    "predicted_type": member_type_index,
                    "predicted_type_name": anomaly_types[member_type_index],
                }
            )
            transition_predictions.append(point)
        event["transition_predictions"] = transition_predictions
        events.append(event)
    return events


def evaluate_event_predictions(
    events: Sequence[Mapping[str, object]],
    anomaly_types: Sequence[str],
    threshold: Optional[float] = None,
) -> dict:
    if not events:
        return {"n_events": 0}
    targets = np.asarray([int(event["target"]) for event in events], dtype=np.int64)
    binary_targets = (targets != 0).astype(np.int64)
    scores = np.asarray(
        [float(event["anomaly_score"]) for event in events], dtype=np.float64
    )
    per_event_thresholds = None
    if threshold is None and all("decision_threshold" in event for event in events):
        per_event_thresholds = np.asarray(
            [float(event["decision_threshold"]) for event in events], dtype=np.float64
        )
        binary_predictions = (scores >= per_event_thresholds).astype(np.int64)
        threshold_mode = "validation_per_fold"
    else:
        threshold_mode = "single"
    if threshold is None and per_event_thresholds is None:
        threshold = select_f1_threshold(binary_targets, scores)
    if per_event_thresholds is None:
        binary_predictions = (scores >= float(threshold)).astype(np.int64)
    tp, fp, fn, tn = binary_counts(binary_targets, binary_predictions)

    probabilities = np.asarray(
        [event["probabilities"] for event in events], dtype=np.float64
    )
    type_predictions = np.zeros(len(events), dtype=np.int64)
    anomalous_predictions = binary_predictions == 1
    if np.any(anomalous_predictions):
        type_predictions[anomalous_predictions] = 1 + np.argmax(
            probabilities[anomalous_predictions, 1:], axis=1
        )
    per_type = {}
    supported_f1 = []
    for type_id, name in enumerate(anomaly_types):
        support = int(np.sum(targets == type_id))
        class_tp = int(np.sum((targets == type_id) & (type_predictions == type_id)))
        class_fp = int(np.sum((targets != type_id) & (type_predictions == type_id)))
        class_fn = support - class_tp
        class_f1 = f1_from_counts(class_tp, class_fp, class_fn)
        per_type[name] = {"support": support, "f1": class_f1}
        if type_id != 0 and support:
            supported_f1.append(class_f1)

    temporal_id = anomaly_types.index("error_temporal")
    temporal_targets = (targets == temporal_id).astype(np.int64)
    temporal_scores = probabilities[:, temporal_id]
    recovery_mask = np.asarray([bool(event.get("recovery", False)) for event in events])
    recovery_fpr = (
        float(binary_predictions[recovery_mask].mean())
        if np.any(recovery_mask)
        else float("nan")
    )
    result = {
        "n_events": len(events),
        "n_anomalies": int(binary_targets.sum()),
        "threshold": float(threshold) if threshold is not None else None,
        "threshold_mode": threshold_mode,
        "anomaly_auprc": average_precision(binary_targets, scores),
        "anomaly_f1": f1_from_counts(tp, fp, fn),
        "precision": float(tp / (tp + fp)) if tp + fp else 0.0,
        "recall": float(tp / (tp + fn)) if tp + fn else 0.0,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "type_macro_f1_supported": (
            float(np.mean(supported_f1)) if supported_f1 else float("nan")
        ),
        "temporal_auprc": average_precision(temporal_targets, temporal_scores),
        "recovery_fpr": recovery_fpr,
        "brier": float(np.mean((scores - binary_targets) ** 2)),
        "ece": expected_calibration_error(binary_targets, scores),
        "per_type": per_type,
    }
    if any("detected" in event for event in events):
        detection_tp = sum(
            bool(event.get("detected", False))
            and not bool(event.get("is_false_positive", False))
            for event in events
        )
        detection_fp = sum(
            bool(event.get("is_false_positive", False)) for event in events
        )
        detection_fn = sum(not bool(event.get("detected", False)) for event in events)
        result["event_detection_precision"] = (
            detection_tp / (detection_tp + detection_fp)
            if detection_tp + detection_fp
            else 0.0
        )
        result["event_detection_recall"] = (
            detection_tp / (detection_tp + detection_fn)
            if detection_tp + detection_fn
            else 0.0
        )
        result["event_detection_f1"] = f1_from_counts(
            detection_tp, detection_fp, detection_fn
        )
    return result
