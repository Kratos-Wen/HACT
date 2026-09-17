#!/usr/bin/env python3
"""Derive a procedure step list from the reference executions of a benchmark.

The construction uses only the reference executions declared in the benchmark
manifest, never the evaluation recordings, and applies one rule:

1. Keep the annotated events of the reference executions whose verb changes the
   state of a component; the auxiliary manipulations ``hold``, ``align``,
   ``adjust``, ``flip``, and ``transfer`` are excluded.
2. Group the remaining events by the object they act on; a group is one
   subprocedure.
3. Order the groups by the median position of the object's first occurrence in
   the reference executions, and order the steps inside a group by the median
   position of the verb's first occurrence.
4. Verbs of one object that occur at the same position in different references
   form one step with alternative realizations.

The result has the same schema as a manually written SOP, so the pipeline reads
it unchanged.  ``expected_tool`` stays unset because the public annotations
carry no tool attribute on action events.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from statistics import median

AUXILIARY_VERBS = ("hold", "align", "adjust", "flip", "transfer")


def _read(path: Path) -> dict:
    with Path(path).open(encoding="utf-8") as handle:
        return json.load(handle)


def _sources(benchmark: Path, fold: int | None) -> tuple[list[dict], str, bool]:
    """Reference executions, or the training executions of one fold."""

    manifest = _read(benchmark / "manifest.json")
    if fold is None:
        references = manifest.get("experts", [])
        if not references:
            raise ValueError(f"{benchmark} declares no reference executions")
        return references, "reference executions", True
    splits = _read(benchmark / "splits.json")["folds"]
    split = next(item for item in splits if int(item["fold"]) == fold)
    training = set(split["train"])
    records = [row for row in manifest["workers"] if str(row["video_id"]) in training]
    if not records:
        raise ValueError(f"fold {fold} has no training executions")
    return records, f"training executions of fold {fold}", False


def derive(
    benchmark: Path,
    procedure_name: str,
    tolerance: float,
    fold: int | None = None,
    minimum_support: float = 0.5,
) -> dict:
    references, source_name, drop_anomalies = _sources(benchmark, fold)
    verb_positions: dict[tuple[str, str], list[float]] = defaultdict(list)
    object_positions: dict[str, list[float]] = defaultdict(list)
    for record in references:
        annotation = _read(Path(record["parsed_annotation_path"]))
        events = sorted(
            (
                event
                for event in annotation["events"]
                if (not drop_anomalies or not event["has_anomaly"])
                and event["verb"] not in AUXILIARY_VERBS
            ),
            key=lambda event: (event["start_frame"], event["end_frame"]),
        )
        if not events:
            continue
        seen_verb: set[tuple[str, str]] = set()
        seen_object: set[str] = set()
        for index, event in enumerate(events):
            position = index / len(events)
            key = (str(event["noun_object_name"]), str(event["verb"]))
            if key not in seen_verb:
                verb_positions[key].append(position)
                seen_verb.add(key)
            if key[0] not in seen_object:
                object_positions[key[0]].append(position)
                seen_object.add(key[0])

    # A step enters the list when it recurs in at least ``minimum_support`` of the
    # source executions; a deviation performed in few executions does not.
    support = max(1, round(minimum_support * len(references)))
    verb_positions = {
        key: values for key, values in verb_positions.items() if len(values) >= support
    }
    object_positions = {
        name: values
        for name, values in object_positions.items()
        if any(key[0] == name for key in verb_positions)
    }
    groups = []
    for object_name in sorted(object_positions, key=lambda name: median(object_positions[name])):
        verbs = [key for key in verb_positions if key[0] == object_name]
        verbs.sort(key=lambda key: median(verb_positions[key]))
        steps: list[dict] = []
        for object_name_, verb in verbs:
            position = median(verb_positions[(object_name_, verb)])
            label = f"{verb}:{object_name_}"
            if steps and abs(position - steps[-1]["_position"]) <= tolerance:
                steps[-1]["step_label"].append(label)
                steps[-1]["_position"] = min(steps[-1]["_position"], position)
                continue
            steps.append({"step_label": [label], "_position": position})
        group = []
        for step_index, step in enumerate(steps):
            labels = step["step_label"]
            group.append(
                {
                    "step_id": step_index,
                    "step_label": labels if len(labels) > 1 else labels[0],
                    "expected_tool": None,
                    "required_objects": [object_name],
                    "penalty": -15,
                }
            )
        groups.append((object_name, group))

    return {
        "procedure_name": procedure_name,
        "source": source_name,
        "anomaly_labels_used": drop_anomalies,
        "minimum_support_fraction": minimum_support,
        "description": (
            f"Derived automatically from the {source_name} of this benchmark: "
            "events are grouped by the object they act on, groups and steps are ordered by "
            "the median position of their first occurrence, and verbs of one object at the "
            f"same position form one step. Auxiliary manipulations ({', '.join(AUXILIARY_VERBS)}) "
            "are excluded. No manual editing."
        ),
        "reference_executions": [str(record["video_id"]) for record in references],
        "group_names": [name for name, _ in groups],
        "steps": [group for _, group in groups],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--procedure-name", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fold", type=int, default=None, help="derive from this fold's training executions instead of the reference executions; anomaly labels are then not used")
    parser.add_argument("--minimum-support", type=float, default=0.5, help="fraction of source executions in which a step must recur")
    parser.add_argument(
        "--position-tolerance",
        type=float,
        default=0.02,
        help="verbs of one object within this normalized distance form one step",
    )
    args = parser.parse_args()
    sop = derive(args.benchmark.resolve(), args.procedure_name, args.position_tolerance, args.fold, args.minimum_support)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(sop, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    steps = sum(len(group) for group in sop["steps"])
    print(
        f"{args.output}: {len(sop['steps'])} subprocedures, {steps} steps, "
        f"from {sop['reference_executions']}"
    )


if __name__ == "__main__":
    main()
