#!/usr/bin/env python3
"""Re-score trained HACT checkpoints while retaining predicted phase scores.

This is inference-only.  It reads cached visual features and checkpoint-owned
splits, models, calibration temperatures, and reference memory.  Ground-truth
annotations are not loaded; the benchmark manifest supplies only feature paths,
frame counts, and FPS metadata.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch


PROJECT = Path(__file__).resolve().parents[1]
for directory in (PROJECT / "pipeline", PROJECT / "utils"):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from bimanual_transition_data import (  # noqa: E402
    ANOMALY_TYPES,
    VideoExample,
    build_frame_targets,
    save_json,
)
from bimanual_transition_metrics import aggregate_transition_predictions  # noqa: E402
from bimanual_transition_model import decode_observed_events  # noqa: E402
from step2_bimanual_transition import (  # noqa: E402
    _transition_batch,
    apply_event_decisions,
    attach_factor_names,
    collect_transition_rows,
    encode_example,
    load_trained_models,
    resolve_device,
)


def _read(path: Path):
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _score_ids(
    video_ids: list[str],
    records: dict[str, dict],
    checkpoint: dict,
    vocab,
    observation,
    transition,
    device: torch.device,
) -> list[dict]:
    reference_memory = {
        key: value.to(device) if value is not None else None
        for key, value in checkpoint["reference_memory"].items()
    }
    use_phase = bool(checkpoint.get("use_phase_transitions", True))
    results = []
    for video_id in video_ids:
        record = records[video_id]
        with np.load(record["feature_path"]) as payload:
            features = np.asarray(payload["features"], dtype=np.float32)
            frame_ids = np.asarray(payload["frame_ids"], dtype=np.int64)
        empty_targets = build_frame_targets([], frame_ids)
        encoded = encode_example(
            observation,
            VideoExample(
                video_id,
                float(record["fps"]),
                features,
                frame_ids,
                [],
                empty_targets,
            ),
            device,
        )
        encoded.events = decode_observed_events(frame_ids, encoded.logits, vocab)
        if not encoded.events:
            continue
        batch = _transition_batch(encoded, encoded.events, use_phase, device)
        rows = collect_transition_rows(
            transition,
            {video_id: batch},
            [],
            checkpoint.get("sop_marks", []),
            memory=reference_memory,
        )
        events = aggregate_transition_predictions(
            rows,
            ANOMALY_TYPES,
            temperature=checkpoint.get("temperature", 1.0),
        )
        # These fields originate from the deliberately empty inference
        # placeholder, not annotations.  Remove them so deployment exports
        # cannot be mistaken for oracle-labelled rows.
        for event in events:
            for key in ("target", "target_name", "recovery"):
                event.pop(key, None)
        apply_event_decisions(events, float(checkpoint.get("threshold", 0.5)))
        attach_factor_names(events, vocab)
        results.extend(events)
    return results


def rescore(
    checkpoint_root: Path,
    benchmark: Path,
    output_root: Path,
    device_name: str,
    seed: int | None = None,
    transfer_test: bool = False,
) -> None:
    """Score every fold checkpoint on the benchmark's videos.

    With ``transfer_test`` the benchmark may replace the test list of each fold
    (a cross-product transfer benchmark whose train/val lists are the original
    fold lists); the frozen models are applied unchanged to the new test videos.
    """
    manifest = _read(benchmark / "manifest.json")
    benchmark_splits = {
        int(row["fold"]): row for row in _read(benchmark / "splits.json")["folds"]
    }
    records = {str(row["video_id"]): row for row in manifest["workers"]}
    device = resolve_device(device_name)
    checkpoint_pattern = (
        f"seed_{seed}/fold_*/model.pt" if seed is not None else "seed_*/fold_*/model.pt"
    )
    checkpoint_paths = sorted(checkpoint_root.glob(checkpoint_pattern))
    if not checkpoint_paths:
        raise FileNotFoundError(
            f"No checkpoints matching {checkpoint_pattern!r} below {checkpoint_root}"
        )
    for checkpoint_path in checkpoint_paths:
        checkpoint, vocab, observation, transition = load_trained_models(
            str(checkpoint_path), device
        )
        split = checkpoint["split"]
        fold = int(checkpoint["fold"])
        frozen_split = benchmark_splits.get(fold)
        if frozen_split is None:
            raise ValueError(f"Checkpoint fold {fold} is absent from benchmark splits")
        for partition in ("train", "val", "test"):
            if transfer_test and partition == "test":
                continue
            if list(split[partition]) != list(frozen_split[partition]):
                raise ValueError(
                    f"Checkpoint/benchmark {partition} mismatch in fold {fold}: "
                    f"{split[partition]} != {frozen_split[partition]}"
                )
        if transfer_test:
            split = {**split, "test": list(frozen_split["test"])}
        relative = checkpoint_path.parent.relative_to(checkpoint_root)
        destination = output_root / relative
        destination.mkdir(parents=True, exist_ok=True)
        validation = _score_ids(
            list(split["val"]),
            records,
            checkpoint,
            vocab,
            observation,
            transition,
            device,
        )
        test = _score_ids(
            list(split["test"]),
            records,
            checkpoint,
            vocab,
            observation,
            transition,
            device,
        )
        training = _score_ids(
            list(split["train"]),
            records,
            checkpoint,
            vocab,
            observation,
            transition,
            device,
        )
        save_json(str(destination / "training_event_predictions.json"), training)
        save_json(str(destination / "validation_event_predictions.json"), validation)
        save_json(str(destination / "predicted_event_predictions.json"), test)
        save_json(
            str(destination / "rescore_manifest.json"),
            {
                "source_checkpoint": str(checkpoint_path.resolve()),
                "ground_truth_inputs": [],
                "training_videos": list(split["train"]),
                "validation_videos": list(split["val"]),
                "test_videos": list(split["test"]),
                "timeline_fields": ["event", "transition_predictions"],
            },
        )
        print(
            f"{relative}: training={len(training)} events, "
            f"validation={len(validation)} events, "
            f"test={len(test)} events"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optionally export only one seed directory (for example, --seed 0).",
    )
    parser.add_argument(
        "--transfer-test",
        action="store_true",
        help="Take each fold's test list from the benchmark (cross-product transfer); "
        "train/val lists must still equal the checkpoint's.",
    )
    args = parser.parse_args()
    rescore(
        args.checkpoint_root.resolve(),
        args.benchmark.resolve(),
        args.output_root.resolve(),
        args.device,
        args.seed,
        args.transfer_test,
    )


if __name__ == "__main__":
    main()
