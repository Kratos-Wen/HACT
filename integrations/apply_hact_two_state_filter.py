#!/usr/bin/env python3
"""Apply HACT's causal normal/error belief filter to phase emissions.

The initial distribution and transition matrix are maximum-likelihood counts
from each fold's training participants.  Recovery annotations are normal under
the binary anomaly target and are therefore included in the normal state.
Validation and test filtering consume predictions only; the filter has no
learned parameter or additive smoothing constant.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


STATES = ("normal", "error")


def _read(path: Path):
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _source_split(source: Path, source_manifest: dict) -> dict[str, list[str]]:
    """Read the fold split recorded with the source checkpoint."""
    manifest_keys = {
        "train": "training_videos",
        "val": "validation_videos",
        "test": "test_videos",
    }
    if all(key in source_manifest for key in manifest_keys.values()):
        return {
            partition: list(source_manifest[key])
            for partition, key in manifest_keys.items()
        }

    checkpoint = Path(str(source_manifest.get("source_checkpoint", "")))
    if not checkpoint.is_absolute():
        checkpoint = (source / checkpoint).resolve()
    split_path = checkpoint.parent / "split.json"
    if not split_path.is_file():
        raise ValueError(
            "Source manifest omits its training split and the checkpoint "
            f"split is unavailable: {split_path}"
        )
    recorded = _read(split_path)
    return {
        partition: list(recorded[partition])
        for partition in ("train", "val", "test")
    }


def _state(event: dict) -> int:
    label = str(event.get("anomaly_label", "normal")).strip().lower()
    return int(label not in {"", "normal", "none", "null", "recovery"})


def _fit_dynamics(video_ids: list[str], records: dict[str, dict]) -> dict:
    initial_counts = np.zeros(2, dtype=np.float64)
    transition_counts = np.zeros((2, 2), dtype=np.float64)
    state_counts = np.zeros(2, dtype=np.float64)
    for video_id in video_ids:
        events = _read(Path(records[video_id]["parsed_annotation_path"]))["events"]
        for hand in ("Left_hand", "Right_hand"):
            sequence = sorted(
                (event for event in events if str(event.get("hand")) == hand),
                key=lambda event: (
                    int(event["start_frame"]),
                    int(event["end_frame"]),
                    str(event.get("event_id", "")),
                ),
            )
            states = [_state(event) for event in sequence]
            for state in states:
                state_counts[state] += 1.0
            if states:
                initial_counts[states[0]] += 1.0
            for previous, current in zip(states, states[1:]):
                transition_counts[previous, current] += 1.0
    if initial_counts.sum() == 0.0 or np.any(transition_counts.sum(axis=1) == 0.0):
        raise ValueError("Training split does not identify both-state dynamics")
    return {
        "states": list(STATES),
        "scope": "hand",
        "training_view": "annotated_events",
        "recovery_mapping": "normal",
        "initial_counts": initial_counts.tolist(),
        "state_counts": state_counts.tolist(),
        "transition_counts": transition_counts.tolist(),
        "initial": (initial_counts / initial_counts.sum()).tolist(),
        "transition": (
            transition_counts / transition_counts.sum(axis=1, keepdims=True)
        ).tolist(),
        "emission_mode": "phase",
        "emission_form": "posterior",
    }


def _emission(score: float) -> np.ndarray:
    value = float(np.clip(score, 1e-8, 1.0 - 1e-8))
    return np.asarray([1.0 - value, value], dtype=np.float64)


def _update(prior: np.ndarray, score: float) -> np.ndarray:
    posterior = prior * _emission(score)
    normalizer = float(posterior.sum())
    return posterior / normalizer if normalizer > 0.0 else prior


def _filter_rows(rows: list[dict], dynamics: dict) -> list[dict]:
    initial = np.asarray(dynamics["initial"], dtype=np.float64)
    transition = np.asarray(dynamics["transition"], dtype=np.float64)
    grouped: dict[tuple[str, int], list[dict]] = {}
    for source in rows:
        if source.get("detected") is False:
            continue
        key = (str(source["video_id"]), int(source.get("hand", -1)))
        grouped.setdefault(key, []).append(source)

    transformed = []
    for key in sorted(grouped):
        posterior = initial.copy()
        sequence = sorted(
            grouped[key],
            key=lambda event: (
                int(event["start_frame"]),
                int(event["end_frame"]),
                str(event.get("event_id", "")),
            ),
        )
        for index, source in enumerate(sequence):
            prior = initial if index == 0 else posterior @ transition
            event = dict(source)
            event["state_prior"] = prior.tolist()
            points = event.get("transition_predictions", [])
            if points:
                filtered_points = []
                phase_prior = prior
                for source_point in sorted(
                    points,
                    key=lambda point: (
                        int(point["frame"]), int(point.get("transition", -1))
                    ),
                ):
                    point = dict(source_point)
                    point_score = float(point["anomaly_score"])
                    posterior = _update(phase_prior, point_score)
                    point["base_anomaly_score"] = point_score
                    point["state_prior"] = phase_prior.tolist()
                    point["state_probabilities"] = posterior.tolist()
                    point["anomaly_score"] = float(posterior[1])
                    filtered_points.append(point)
                    phase_prior = posterior
                event["transition_predictions"] = filtered_points
                base_score = float(source["anomaly_score"])
            else:
                base_score = float(source["anomaly_score"])
                posterior = _update(prior, base_score)
            event["base_anomaly_score"] = base_score
            event["state_probabilities"] = posterior.tolist()
            event["anomaly_score"] = float(posterior[1])
            if "probabilities" in event:
                probabilities = np.asarray(event["probabilities"], dtype=np.float64)
                subtype_mass = probabilities[1:]
                subtype_total = float(subtype_mass.sum())
                if subtype_total > 0.0:
                    subtype_mass = subtype_mass / subtype_total * posterior[1]
                event["probabilities"] = np.r_[
                    1.0 - posterior[1], subtype_mass
                ].tolist()
            transformed.append(event)
    return transformed


def apply_filter(prediction_root: Path, benchmark: Path, output_root: Path, seed: int = 0) -> None:
    manifest = _read(benchmark / "manifest.json")
    records = {str(row["video_id"]): row for row in manifest["workers"]}
    folds = _read(benchmark / "splits.json")["folds"]
    for split in folds:
        fold = int(split["fold"])
        source = prediction_root / f"seed_{seed}" / f"fold_{fold}"
        destination = output_root / f"seed_{seed}" / f"fold_{fold}"
        destination.mkdir(parents=True, exist_ok=True)
        source_manifest = _read(source / "rescore_manifest.json")
        recorded_split = _source_split(source, source_manifest)
        for partition in ("train", "val", "test"):
            if recorded_split[partition] != list(split[partition]):
                raise ValueError(
                    f"Prediction/benchmark {partition} mismatch in fold {fold}"
                )
        dynamics = _fit_dynamics(list(split["train"]), records)
        validation = _filter_rows(
            _read(source / "validation_event_predictions.json"), dynamics
        )
        test = _filter_rows(
            _read(source / "predicted_event_predictions.json"), dynamics
        )
        for name, payload in (
            ("validation_event_predictions.json", validation),
            ("predicted_event_predictions.json", test),
            ("state_dynamics.json", dynamics),
            (
                "rescore_manifest.json",
                {
                    **source_manifest,
                    "training_videos": recorded_split["train"],
                    "validation_videos": recorded_split["val"],
                    "test_videos": recorded_split["test"],
                    "source_prediction_directory": str(source.resolve()),
                    "causal_filter": "normal_error_count_mle",
                },
            ),
        ):
            (destination / name).write_text(
                json.dumps(payload, indent=2) + "\n", encoding="utf-8"
            )
        print(f"fold {fold}: validation={len(validation)}, test={len(test)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prediction-root", type=Path, required=True)
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0,
                        help="which seed subdirectory of the rescored root to filter")
    args = parser.parse_args()
    apply_filter(
        args.prediction_root.resolve(),
        args.benchmark.resolve(),
        args.output_root.resolve(),
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
