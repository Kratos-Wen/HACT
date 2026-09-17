"""Deterministic SOP-aware scoring for novice procedural execution."""

from __future__ import annotations

import itertools
from collections import defaultdict
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np


def _expanded_groups(requirements: Mapping[str, object]) -> List[List[dict]]:
    expanded: List[List[dict]] = []
    for group in requirements.get("groups", []):
        chain = []
        for step in group.get("steps", []):
            minimum = int(step.get("min_repeats", 1))
            maximum = int(step.get("max_repeats", minimum))
            for repeat_index in range(maximum):
                chain.append(
                    {
                        **dict(step),
                        "group_name": str(group.get("name", "")),
                        "repeat_index": repeat_index,
                        "required": repeat_index < minimum,
                        "requirement_id": (
                            f"g{int(step['group_index'])}:"
                            f"s{int(step['step_index'])}:r{repeat_index}"
                        ),
                    }
                )
        expanded.append(chain)
    return expanded


def _compatibility_index(
    groups: Sequence[Sequence[Mapping[str, object]]],
) -> List[Dict[Tuple[int, int], Tuple[int, ...]]]:
    result = []
    for chain in groups:
        positions: Dict[Tuple[int, int], set] = defaultdict(set)
        for requirement_index, requirement in enumerate(chain):
            for alternative in requirement.get("alternatives", []):
                key = (int(alternative["verb"]), int(alternative["part"]))
                positions[key].add(requirement_index)
        result.append(
            {key: tuple(sorted(value)) for key, value in positions.items()}
        )
    return result


def _quality(event: Mapping[str, object]) -> float:
    if "quality" in event:
        value = float(event["quality"])
    else:
        value = 1.0 - float(event.get("anomaly_score", 0.0))
    return float(np.clip(value, 0.0, 1.0))


def _better(first: tuple, second: tuple) -> bool:
    """Compare paths by score, then completed steps, then stable event order."""

    first_key = first[:3]
    second_key = second[:3]
    return first_key > second_key


def _consume_event(
    states: Dict[Tuple[int, ...], tuple],
    event_index: int,
    event: Mapping[str, object],
    groups: Sequence[Sequence[Mapping[str, object]]],
    compatibility: Sequence[Mapping[Tuple[int, int], Sequence[int]]],
) -> Dict[Tuple[int, ...], tuple]:
    updated = dict(states)
    contribution = _quality(event)
    semantic_key = (int(event.get("verb", -1)), int(event.get("part", -1)))
    for state, path in states.items():
        for group_index, chain in enumerate(groups):
            for requirement_index in compatibility[group_index].get(semantic_key, ()):
                if requirement_index < state[group_index]:
                    continue
                requirement = chain[requirement_index]
                next_state = list(state)
                next_state[group_index] = requirement_index + 1
                key = tuple(next_state)
                required = bool(requirement["required"])
                candidate = (
                    path[0] + (contribution if required else 0.0),
                    path[1] + int(required),
                    path[2] + int(not required),
                    path[3] + ((event_index, group_index, requirement_index),),
                )
                previous = updated.get(key)
                if previous is None or _better(candidate, previous):
                    updated[key] = candidate
    return updated


def _event_time_groups(events: Sequence[Mapping[str, object]]) -> List[List[int]]:
    grouped: Dict[int, List[int]] = defaultdict(list)
    for index, event in enumerate(events):
        grouped[int(event.get("start_frame", index))].append(index)
    return [grouped[frame] for frame in sorted(grouped)]


def score_procedure(
    events: Sequence[Mapping[str, object]],
    requirements: Optional[Mapping[str, object]],
) -> dict:
    """Score a detected event sequence against parallel SOP prerequisite chains.

    The dynamic program permits arbitrary interleaving of top-level SOP groups
    while preserving order inside each group.  The final score has no tuned
    component weights: required steps contribute their calibrated correctness,
    while SOP-authorized repeats are absorbed as optional corrective actions.
    """

    if not requirements:
        return {
            "available": False,
            "procedure_score": None,
            "reason": "No SOP requirements were stored with the model",
        }

    groups = _expanded_groups(requirements)
    compatibility = _compatibility_index(groups)
    required_count = sum(
        int(requirement["required"])
        for group in groups
        for requirement in group
    )
    ordered_events = sorted(
        (dict(event) for event in events),
        key=lambda event: (
            int(event.get("start_frame", 0)),
            int(event.get("end_frame", 0)),
            int(event.get("hand", -1)),
            str(event.get("event_id", "")),
        ),
    )
    initial_state = tuple(0 for _ in groups)
    states: Dict[Tuple[int, ...], tuple] = {
        initial_state: (0.0, 0, 0, tuple())
    }

    # Events with the same start time are an unordered bimanual set.  In the
    # expected two-hand case, trying both serializations is exact and cheap.
    for indices in _event_time_groups(ordered_events):
        orderings = (
            itertools.permutations(indices)
            if len(indices) <= 4
            else (tuple(indices),)
        )
        merged: Dict[Tuple[int, ...], tuple] = {}
        for ordering in orderings:
            candidate_states = states
            for event_index in ordering:
                candidate_states = _consume_event(
                    candidate_states,
                    event_index,
                    ordered_events[event_index],
                    groups,
                    compatibility,
                )
            for state, path in candidate_states.items():
                previous = merged.get(state)
                if previous is None or _better(path, previous):
                    merged[state] = path
        states = merged

    best = max(states.values(), key=lambda item: item[:3])
    quality_sum, required_matches, optional_matches, path = best
    matched_event_indices = {item[0] for item in path}
    matched_requirement_ids = {
        groups[group_index][requirement_index]["requirement_id"]
        for _, group_index, requirement_index in path
    }
    matched_steps = []
    for event_index, group_index, requirement_index in path:
        event = ordered_events[event_index]
        requirement = groups[group_index][requirement_index]
        matched_steps.append(
            {
                "requirement_id": requirement["requirement_id"],
                "group_index": int(requirement["group_index"]),
                "group_name": requirement["group_name"],
                "step_index": int(requirement["step_index"]),
                "repeat_index": int(requirement["repeat_index"]),
                "required": bool(requirement["required"]),
                "accepted_labels": [
                    alternative.get("label")
                    for alternative in requirement.get("alternatives", [])
                ],
                "event_id": str(event.get("event_id", event_index)),
                "observed_label": (
                    f"{event.get('verb_name')}:{event.get('part_name')}"
                    if event.get("verb_name") is not None
                    and event.get("part_name") is not None
                    else None
                ),
                "event_quality": _quality(event),
                "anomaly_score": float(event.get("anomaly_score", 0.0)),
            }
        )
    missing_steps = []
    for chain in groups:
        for requirement in chain:
            if (
                requirement["required"]
                and requirement["requirement_id"] not in matched_requirement_ids
            ):
                missing_steps.append(
                    {
                        "requirement_id": requirement["requirement_id"],
                        "group_index": int(requirement["group_index"]),
                        "group_name": requirement["group_name"],
                        "step_index": int(requirement["step_index"]),
                        "repeat_index": int(requirement["repeat_index"]),
                        "accepted_labels": [
                            alternative.get("label")
                            for alternative in requirement.get("alternatives", [])
                        ],
                    }
                )
    extra_events = [
        {
            "event_id": str(event.get("event_id", index)),
            "start_frame": int(event.get("start_frame", 0)),
            "end_frame": int(event.get("end_frame", 0)),
            "verb": int(event.get("verb", -1)),
            "part": int(event.get("part", -1)),
            "verb_name": event.get("verb_name"),
            "part_name": event.get("part_name"),
            "tool_name": event.get("tool_name"),
            "anomaly_score": float(event.get("anomaly_score", 0.0)),
        }
        for index, event in enumerate(ordered_events)
        if index not in matched_event_indices
    ]
    matched_count = int(required_matches)
    effective_observed_count = len(ordered_events) - int(optional_matches)
    denominator = max(required_count, effective_observed_count, 1)
    execution_quality = quality_sum / matched_count if matched_count else 0.0
    uncertainty = (
        float(
            np.mean(
                [
                    1.0 - abs(2.0 * float(event.get("anomaly_score", 0.0)) - 1.0)
                    for event in ordered_events
                ]
            )
        )
        if ordered_events
        else 0.0
    )
    return {
        "available": True,
        "procedure_name": str(requirements.get("procedure_name", "procedure")),
        "procedure_score": 100.0 * quality_sum / denominator,
        "score_definition": (
            "100 * required-step calibrated correctness / "
            "max(required steps, observed non-optional events)"
        ),
        "required_steps": required_count,
        "observed_events": len(ordered_events),
        "matched_steps_count": matched_count,
        "matched_optional_repeats": int(optional_matches),
        "missing_steps_count": len(missing_steps),
        "extra_events_count": len(extra_events),
        "completion": matched_count / required_count if required_count else 0.0,
        "event_precision": (
            (matched_count + int(optional_matches)) / len(ordered_events)
            if ordered_events
            else 0.0
        ),
        "execution_quality": execution_quality,
        "sequence_coverage": matched_count / denominator,
        "mean_decision_uncertainty": uncertainty,
        "matched_steps": matched_steps,
        "missing_steps": missing_steps,
        "extra_events": extra_events,
    }


def _average_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="stable")
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1)
        start = end
    return ranks


def _correlation(first: np.ndarray, second: np.ndarray) -> float:
    if len(first) < 2 or np.std(first) == 0 or np.std(second) == 0:
        return float("nan")
    return float(np.corrcoef(first, second)[0, 1])


def evaluate_procedure_scores(rows: Sequence[Mapping[str, object]]) -> dict:
    if not rows:
        return {"n_videos": 0}
    predicted = np.asarray(
        [float(row["predicted_score"]) for row in rows], dtype=np.float64
    )
    reference = np.asarray(
        [float(row["reference_score"]) for row in rows], dtype=np.float64
    )
    residual = predicted - reference
    return {
        "n_videos": len(rows),
        "mae": float(np.mean(np.abs(residual))),
        "rmse": float(np.sqrt(np.mean(residual**2))),
        "bias": float(np.mean(residual)),
        "pearson": _correlation(predicted, reference),
        "spearman": _correlation(_average_ranks(predicted), _average_ranks(reference)),
    }
