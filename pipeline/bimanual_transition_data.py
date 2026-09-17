"""Data contracts for expert-conditioned bimanual transition modelling.

Every annotated event contributes its observed verb, part, tool and phase
targets.  Correctness is represented by a separate anomaly target.
"""

from __future__ import annotations

import csv
import json
import os
import random
from dataclasses import asdict, dataclass
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np


HANDS: Tuple[str, ...] = ("left", "right")
PHASE_STATES: Tuple[str, ...] = ("idle", "approach", "interaction")
TRANSITION_TYPES: Tuple[str, ...] = ("start", "onset", "end")
ANOMALY_TYPES: Tuple[str, ...] = (
    "normal",
    "error_temporal",
    "error_spatial",
    "error_handling",
    "error_wrong_part",
    "error_wrong_tool",
    "error_procedural",
)
RECOVERY_LABEL = "recovery"
NONE_TOKEN = "<none>"
UNK_TOKEN = "<unk>"


def normalize_hand(value: str) -> str:
    hand = str(value or "").strip().lower().replace("-", "_")
    if hand in {"left", "left_hand", "lefthand", "l"}:
        return "left"
    if hand in {"right", "right_hand", "righthand", "r"}:
        return "right"
    raise ValueError(f"Unknown hand label: {value!r}")


def canonical_anomaly_label(event: Mapping[str, object]) -> Tuple[str, bool]:
    """Return the seven-class target and whether this is a recovery event.

    Recovery is a valid corrective transition, not a seventh anomaly.  It is
    mapped to ``normal`` for correctness supervision and retained through a
    separate binary mark so that it changes the model history and can be
    evaluated for false positives.
    """

    raw = str(event.get("anomaly_label", "normal") or "normal").strip().lower()
    if raw == RECOVERY_LABEL:
        return "normal", True
    if raw in {"", "none", "null"}:
        raw = "normal"
    if raw not in ANOMALY_TYPES:
        raise ValueError(
            f"Unsupported anomaly label {raw!r}; expected one of "
            f"{list(ANOMALY_TYPES)} or {RECOVERY_LABEL!r}"
        )
    has_anomaly = bool(event.get("has_anomaly", raw != "normal"))
    if raw == "normal" and has_anomaly:
        raise ValueError("Event has_anomaly=true but anomaly_label='normal'")
    return raw, False


def _clean_factor(value: object) -> str:
    value = str(value or "").lstrip("\ufeff").strip()
    return value if value else NONE_TOKEN


@dataclass(frozen=True)
class FactorVocabulary:
    verbs: Tuple[str, ...]
    parts: Tuple[str, ...]
    tools: Tuple[str, ...]

    @staticmethod
    def _make(values: Iterable[str]) -> Tuple[str, ...]:
        body = sorted({v for v in values if v not in {NONE_TOKEN, UNK_TOKEN, ""}})
        return (NONE_TOKEN, UNK_TOKEN, *body)

    @classmethod
    def build(
        cls,
        annotation_paths: Sequence[str],
        sop_path: Optional[str] = None,
    ) -> "FactorVocabulary":
        verbs: List[str] = []
        parts: List[str] = []
        tools: List[str] = []
        for path in annotation_paths:
            with open(path, encoding="utf-8") as f:
                annotation = json.load(f)
            for event in annotation.get("events", []):
                verbs.append(_clean_factor(event.get("verb")))
                parts.append(
                    _clean_factor(
                        event.get("target_object_name") or event.get("noun_object_name")
                    )
                )
                tools.append(_clean_factor(event.get("tool_object_name")))

        if sop_path and os.path.exists(sop_path):
            for step in load_sop_marks(sop_path):
                verbs.append(step["verb"])
                parts.append(step["part"])
                tools.append(step["tool"])

        return cls(cls._make(verbs), cls._make(parts), cls._make(tools))

    @classmethod
    def build_from_ontology(
        cls,
        verbs_path: str,
        nouns_path: str,
        ontology_path: Optional[str] = None,
        objects_path: Optional[str] = None,
        sop_path: Optional[str] = None,
    ) -> "FactorVocabulary":
        """Build a fixed task vocabulary without reading worker annotations.

        Parts and tools share the task's object ontology.  Keeping separate
        output heads still lets the model learn their different roles, while
        using one closed object inventory prevents fold-dependent dimensions.
        """

        if not verbs_path or not os.path.exists(verbs_path):
            raise FileNotFoundError(f"Fixed verb vocabulary not found: {verbs_path}")
        if not nouns_path or not os.path.exists(nouns_path):
            raise FileNotFoundError(f"Fixed object vocabulary not found: {nouns_path}")

        def read_lines(path: str) -> List[str]:
            with open(path, encoding="utf-8-sig") as f:
                return [_clean_factor(line) for line in f if line.strip()]

        verbs = read_lines(verbs_path)
        objects = read_lines(nouns_path)

        if ontology_path:
            if not os.path.exists(ontology_path):
                raise FileNotFoundError(f"Verb-object ontology not found: {ontology_path}")
            with open(ontology_path, encoding="utf-8-sig", newline="") as f:
                for row in csv.DictReader(f):
                    if str(row.get("allowed", "1")).strip() not in {"0", "false", "False"}:
                        verbs.append(_clean_factor(row.get("verb")))
                        objects.append(_clean_factor(row.get("noun")))

        if objects_path:
            if not os.path.exists(objects_path):
                raise FileNotFoundError(f"Object-state ontology not found: {objects_path}")
            with open(objects_path, encoding="utf-8-sig") as f:
                payload = json.load(f)
            for item in payload.get("objects", []):
                objects.append(_clean_factor(item.get("class_name")))

        if sop_path:
            if not os.path.exists(sop_path):
                raise FileNotFoundError(f"SOP not found: {sop_path}")
            for mark in load_sop_marks(sop_path):
                verbs.append(mark["verb"])
                objects.extend((mark["part"], mark["tool"]))

        return cls(cls._make(verbs), cls._make(objects), cls._make(objects))

    def _items(self, factor: str) -> Tuple[str, ...]:
        if factor == "verb":
            return self.verbs
        if factor == "part":
            return self.parts
        if factor == "tool":
            return self.tools
        raise KeyError(factor)

    def encode(self, factor: str, value: object) -> int:
        items = self._items(factor)
        cleaned = _clean_factor(value)
        try:
            return items.index(cleaned)
        except ValueError:
            return items.index(UNK_TOKEN)

    def decode(self, factor: str, index: int) -> str:
        return self._items(factor)[int(index)]

    def size(self, factor: str) -> int:
        return len(self._items(factor))

    def contains(self, factor: str, value: object) -> bool:
        return _clean_factor(value) in self._items(factor)

    def to_dict(self) -> dict:
        return {
            "verbs": list(self.verbs),
            "parts": list(self.parts),
            "tools": list(self.tools),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Sequence[str]]) -> "FactorVocabulary":
        return cls(
            tuple(payload["verbs"]), tuple(payload["parts"]), tuple(payload["tools"])
        )


@dataclass(frozen=True)
class AnnotatedEvent:
    event_id: str
    hand: int
    start_frame: int
    onset_frame: int
    end_frame: int
    verb: int
    part: int
    tool: int
    anomaly: int
    recovery: bool

    @property
    def is_anomaly(self) -> bool:
        return self.anomaly != 0


@dataclass(frozen=True)
class TransitionRecord:
    video_id: str
    event_id: str
    event_start_frame: int
    event_end_frame: int
    frame: int
    feature_index: int
    pool_start_index: int
    pool_end_index: int
    hand: int
    transition: int
    verb: int
    part: int
    tool: int
    anomaly: int
    recovery: bool
    event_weight: float
    delta_global: float
    delta_hand: float
    delta_cross: float
    has_prev_global: bool
    has_prev_hand: bool
    has_prev_cross: bool
    time_group_weight: float


@dataclass
class FrameTargets:
    verb: np.ndarray  # [2, T]
    part: np.ndarray  # [2, T]
    tool: np.ndarray  # [2, T]
    phase: np.ndarray  # [2, T]
    overlap_count: int = 0


@dataclass
class VideoExample:
    video_id: str
    fps: float
    features: np.ndarray
    frame_ids: np.ndarray
    events: List[AnnotatedEvent]
    targets: FrameTargets


def annotation_path(annotations_dir: str, video_id: str) -> str:
    return os.path.join(annotations_dir, f"{video_id}_parsed_annotations.json")


def feature_path(features_dir: str, video_id: str) -> str:
    return os.path.join(features_dir, f"{video_id}_features.npz")


def discover_video_ids(features_dir: str, annotations_dir: str) -> List[str]:
    if not os.path.isdir(features_dir) or not os.path.isdir(annotations_dir):
        return []
    feature_ids = {
        name[: -len("_features.npz")]
        for name in os.listdir(features_dir)
        if name.endswith("_features.npz")
    }
    annotation_ids = {
        name[: -len("_parsed_annotations.json")]
        for name in os.listdir(annotations_dir)
        if name.endswith("_parsed_annotations.json")
    }
    return sorted(feature_ids & annotation_ids)


def load_events(
    path: str, vocab: FactorVocabulary
) -> Tuple[float, List[AnnotatedEvent]]:
    with open(path, encoding="utf-8") as f:
        annotation = json.load(f)
    fps = float(annotation.get("fps", 25.0) or 25.0)
    if fps <= 0:
        raise ValueError(f"Invalid fps={fps} in {path}")

    events: List[AnnotatedEvent] = []
    for raw in annotation.get("events", []):
        start = int(raw["start_frame"])
        onset = int(raw["contact_onset_frame"])
        end = int(raw["end_frame"])
        if not start <= onset <= end:
            raise ValueError(
                f"Invalid event boundaries in {path}: start={start}, onset={onset}, end={end}"
            )
        anomaly_name, recovery = canonical_anomaly_label(raw)
        events.append(
            AnnotatedEvent(
                event_id=str(raw.get("event_id", len(events))),
                hand=HANDS.index(normalize_hand(str(raw.get("hand", "")))),
                start_frame=start,
                onset_frame=onset,
                end_frame=end,
                verb=vocab.encode("verb", raw.get("verb")),
                part=vocab.encode(
                    "part", raw.get("target_object_name") or raw.get("noun_object_name")
                ),
                tool=vocab.encode("tool", raw.get("tool_object_name")),
                anomaly=ANOMALY_TYPES.index(anomaly_name),
                recovery=recovery,
            )
        )
    events.sort(
        key=lambda e: (e.start_frame, e.onset_frame, e.end_frame, e.hand, e.event_id)
    )
    return fps, events


def audit_annotation_vocabulary(
    annotation_paths: Sequence[str], vocab: FactorVocabulary
) -> Dict[str, Dict[str, int]]:
    """Report labels outside the fixed ontology without adapting to them."""

    unknown: Dict[str, Dict[str, int]] = {
        "verb": {},
        "part": {},
        "tool": {},
    }
    for path in annotation_paths:
        with open(path, encoding="utf-8-sig") as f:
            annotation = json.load(f)
        for event in annotation.get("events", []):
            values = {
                "verb": event.get("verb"),
                "part": event.get("target_object_name")
                or event.get("noun_object_name"),
                "tool": event.get("tool_object_name"),
            }
            for factor, value in values.items():
                cleaned = _clean_factor(value)
                if not vocab.contains(factor, cleaned):
                    unknown[factor][cleaned] = unknown[factor].get(cleaned, 0) + 1
    return unknown


def nearest_frame_index(frame_ids: np.ndarray, frame: int) -> int:
    if len(frame_ids) == 0:
        raise ValueError("frame_ids is empty")
    pos = int(np.searchsorted(frame_ids, frame))
    if pos <= 0:
        return 0
    if pos >= len(frame_ids):
        return len(frame_ids) - 1
    before = int(frame_ids[pos - 1])
    after = int(frame_ids[pos])
    return pos - 1 if abs(frame - before) <= abs(after - frame) else pos


def build_frame_targets(
    events: Sequence[AnnotatedEvent],
    frame_ids: np.ndarray,
) -> FrameTargets:
    """Build observed per-hand targets without removing anomalous events."""

    shape = (len(HANDS), len(frame_ids))
    verb = np.zeros(shape, dtype=np.int64)
    part = np.zeros(shape, dtype=np.int64)
    tool = np.zeros(shape, dtype=np.int64)
    phase = np.zeros(shape, dtype=np.int64)
    occupied = np.zeros(shape, dtype=np.bool_)
    overlap_count = 0

    for event in events:
        indices = np.flatnonzero(
            (frame_ids >= event.start_frame) & (frame_ids <= event.end_frame)
        )
        if not len(indices):
            continue
        hand = event.hand
        overlap_count += int(occupied[hand, indices].sum())
        occupied[hand, indices] = True
        verb[hand, indices] = event.verb
        part[hand, indices] = event.part
        tool[hand, indices] = event.tool
        phase[hand, indices] = np.where(
            frame_ids[indices] < event.onset_frame,
            PHASE_STATES.index("approach"),
            PHASE_STATES.index("interaction"),
        )

    return FrameTargets(
        verb=verb, part=part, tool=tool, phase=phase, overlap_count=overlap_count
    )


def load_video_example(
    video_id: str,
    features_dir: str,
    annotations_dir: str,
    vocab: FactorVocabulary,
) -> VideoExample:
    with np.load(feature_path(features_dir, video_id)) as data:
        features = np.asarray(data["features"], dtype=np.float32)
        frame_ids = np.asarray(data["frame_ids"], dtype=np.int64)
    if features.ndim != 2 or frame_ids.ndim != 1 or len(features) != len(frame_ids):
        raise ValueError(
            f"Malformed feature cache for {video_id}: {features.shape}, {frame_ids.shape}"
        )
    if len(frame_ids) > 1 and np.any(np.diff(frame_ids) <= 0):
        raise ValueError(f"frame_ids must be strictly increasing for {video_id}")
    fps, events = load_events(annotation_path(annotations_dir, video_id), vocab)
    targets = build_frame_targets(events, frame_ids)
    return VideoExample(video_id, fps, features, frame_ids, events, targets)


def build_transitions(
    video_id: str,
    fps: float,
    frame_ids: np.ndarray,
    events: Sequence[AnnotatedEvent],
    use_phase_transitions: bool = True,
) -> List[TransitionRecord]:
    """Merge both hands into a timestamped marked-transition sequence.

    Simultaneous transitions share one global waiting time.  Their
    ``time_group_weight`` values sum to one so that a bimanual transition is
    not counted twice in the temporal likelihood.
    """

    raw: List[dict] = []
    marks = (("start", "start_frame"), ("onset", "onset_frame"), ("end", "end_frame"))
    if not use_phase_transitions:
        marks = (("onset", "onset_frame"),)

    transitions_per_event = float(len(marks))
    for event in events:
        for transition_name, field in marks:
            frame = int(getattr(event, field))
            if transition_name == "start":
                pool_start_frame, pool_end_frame = event.start_frame, event.onset_frame
            elif transition_name == "end":
                pool_start_frame, pool_end_frame = event.onset_frame, event.end_frame
            else:
                pool_start_frame, pool_end_frame = event.start_frame, event.end_frame
            raw.append(
                {
                    "video_id": video_id,
                    "event_id": event.event_id,
                    "event_start_frame": event.start_frame,
                    "event_end_frame": event.end_frame,
                    "frame": frame,
                    "feature_index": nearest_frame_index(frame_ids, frame),
                    "pool_start_index": nearest_frame_index(
                        frame_ids, pool_start_frame
                    ),
                    "pool_end_index": nearest_frame_index(frame_ids, pool_end_frame),
                    "hand": event.hand,
                    "transition": TRANSITION_TYPES.index(transition_name),
                    "verb": event.verb,
                    "part": event.part,
                    "tool": event.tool,
                    "anomaly": event.anomaly,
                    "recovery": event.recovery,
                    "event_weight": 1.0 / transitions_per_event,
                }
            )
    raw.sort(key=lambda x: (x["frame"], x["transition"], x["hand"], x["event_id"]))

    group_sizes: Dict[int, int] = {}
    for item in raw:
        group_sizes[item["frame"]] = group_sizes.get(item["frame"], 0) + 1

    previous_global_by_frame: Dict[int, Optional[int]] = {}
    prior: Optional[int] = None
    for frame in sorted(group_sizes):
        previous_global_by_frame[frame] = prior
        prior = frame

    previous_hand_by_item: Dict[Tuple[int, int], Optional[int]] = {}
    for hand in range(len(HANDS)):
        hand_frames = sorted(
            {int(item["frame"]) for item in raw if int(item["hand"]) == hand}
        )
        prior = None
        for frame in hand_frames:
            previous_hand_by_item[(hand, frame)] = prior
            prior = frame

    previous_cross_by_item: Dict[Tuple[int, int], Optional[int]] = {}
    frames_by_hand = {
        hand: sorted(
            {int(item["frame"]) for item in raw if int(item["hand"]) == hand}
        )
        for hand in range(len(HANDS))
    }
    for hand in range(len(HANDS)):
        other_frames = frames_by_hand[1 - hand]
        for frame in frames_by_hand[hand]:
            position = int(np.searchsorted(other_frames, frame, side="left"))
            previous_cross_by_item[(hand, frame)] = (
                other_frames[position - 1] if position else None
            )

    result: List[TransitionRecord] = []
    resolution = 1.0 / fps
    for item in raw:
        frame = int(item["frame"])
        hand = int(item["hand"])
        previous_global = previous_global_by_frame[frame]
        previous_hand = previous_hand_by_item[(hand, frame)]
        previous_cross = previous_cross_by_item[(hand, frame)]
        delta_global = (
            resolution
            if previous_global is None
            else max(resolution, (frame - int(previous_global)) / fps)
        )
        delta_hand = (
            resolution
            if previous_hand is None
            else max(resolution, (frame - int(previous_hand)) / fps)
        )
        delta_cross = (
            resolution
            if previous_cross is None
            else max(resolution, (frame - int(previous_cross)) / fps)
        )
        result.append(
            TransitionRecord(
                **item,
                delta_global=float(delta_global),
                delta_hand=float(delta_hand),
                delta_cross=float(delta_cross),
                has_prev_global=previous_global is not None,
                has_prev_hand=previous_hand is not None,
                has_prev_cross=previous_cross is not None,
                time_group_weight=1.0 / group_sizes[frame],
            )
        )
    return result


def load_sop_marks(sop_path: str) -> List[dict]:
    """Expand each acceptable SOP action into one factorized memory mark."""

    with open(sop_path, encoding="utf-8") as f:
        sop = json.load(f)
    marks: List[dict] = []
    for group_index, group in enumerate(sop.get("steps", [])):
        for step_index, step in enumerate(group):
            labels = step.get("step_label", [])
            if isinstance(labels, str):
                labels = [labels]
            for label in labels:
                verb, separator, part = str(label).partition(":")
                if not separator:
                    continue
                marks.append(
                    {
                        "verb": _clean_factor(verb),
                        "part": _clean_factor(part),
                        "tool": _clean_factor(step.get("expected_tool")),
                        "group_index": group_index,
                        "step_index": step_index,
                    }
                )
    return marks


def load_sop_requirements(sop_path: str) -> dict:
    """Load the prerequisite-chain structure used by procedure scoring."""

    with open(sop_path, encoding="utf-8-sig") as f:
        sop = json.load(f)
    group_names = list(sop.get("group_names", []))
    groups = []
    for group_index, raw_group in enumerate(sop.get("steps", [])):
        steps = []
        for step_index, raw_step in enumerate(raw_group):
            labels = raw_step.get("step_label", [])
            if isinstance(labels, str):
                labels = [labels]
            alternatives = []
            for label in labels:
                verb, separator, part = str(label).partition(":")
                if not separator:
                    raise ValueError(
                        f"Invalid SOP label {label!r} at group {group_index}, "
                        f"step {step_index}; expected verb:part"
                    )
                alternatives.append(
                    {"verb": _clean_factor(verb), "part": _clean_factor(part)}
                )
            if not alternatives:
                raise ValueError(
                    f"SOP group {group_index}, step {step_index} has no valid labels"
                )
            minimum = int(raw_step.get("min_repeats", 1) or 1)
            maximum = int(raw_step.get("max_repeats", minimum) or minimum)
            if minimum < 1 or maximum < minimum:
                raise ValueError(
                    f"Invalid repeat range [{minimum}, {maximum}] at SOP group "
                    f"{group_index}, step {step_index}"
                )
            steps.append(
                {
                    "group_index": group_index,
                    "step_index": step_index,
                    "step_id": int(raw_step.get("step_id", step_index)),
                    "alternatives": alternatives,
                    "tool": _clean_factor(raw_step.get("expected_tool")),
                    "min_repeats": minimum,
                    "max_repeats": maximum,
                }
            )
        groups.append(
            {
                "group_index": group_index,
                "name": (
                    str(group_names[group_index])
                    if group_index < len(group_names)
                    else f"group_{group_index}"
                ),
                "steps": steps,
            }
        )
    return {
        "procedure_name": str(sop.get("procedure_name", "procedure")),
        "groups": groups,
    }


def encode_sop_requirements(
    sop_path: Optional[str], vocab: FactorVocabulary
) -> Optional[dict]:
    if not sop_path:
        return None
    if not os.path.exists(sop_path):
        raise FileNotFoundError(f"SOP not found: {sop_path}")
    requirements = load_sop_requirements(sop_path)
    encoded_groups = []
    for group in requirements["groups"]:
        encoded_steps = []
        for step in group["steps"]:
            encoded_steps.append(
                {
                    **{key: step[key] for key in (
                        "group_index",
                        "step_index",
                        "step_id",
                        "min_repeats",
                        "max_repeats",
                    )},
                    "alternatives": [
                        {
                            "verb": vocab.encode("verb", alternative["verb"]),
                            "part": vocab.encode("part", alternative["part"]),
                            "label": (
                                f"{alternative['verb']}:{alternative['part']}"
                            ),
                        }
                        for alternative in step["alternatives"]
                    ],
                    "tool": vocab.encode("tool", step["tool"]),
                    "tool_name": step["tool"],
                }
            )
        encoded_groups.append({**group, "steps": encoded_steps})
    return {
        "procedure_name": requirements["procedure_name"],
        "groups": encoded_groups,
    }


def encode_sop_marks(sop_path: Optional[str], vocab: FactorVocabulary) -> List[dict]:
    if not sop_path or not os.path.exists(sop_path):
        return []
    encoded = []
    for mark in load_sop_marks(sop_path):
        encoded.append(
            {
                "verb": vocab.encode("verb", mark["verb"]),
                "part": vocab.encode("part", mark["part"]),
                "tool": vocab.encode("tool", mark["tool"]),
                "group_index": int(mark["group_index"]),
                "step_index": int(mark["step_index"]),
            }
        )
    return encoded


def load_group_map(
    video_ids: Sequence[str], groups_json: Optional[str] = None
) -> Dict[str, str]:
    """Resolve participant IDs, defaulting to the prefix before ``_``."""

    explicit: Mapping[str, object] = {}
    if groups_json:
        with open(groups_json, encoding="utf-8") as f:
            payload = json.load(f)
        explicit = payload.get("video_to_group", payload)
        missing = sorted(set(video_ids) - set(explicit))
        if missing:
            raise ValueError(
                f"Participant map {groups_json} is missing {len(missing)} video(s): {missing}"
            )
    result = {}
    for video_id in video_ids:
        if video_id in explicit:
            group = str(explicit[video_id]).strip()
            if not group:
                raise ValueError(f"Participant map has an empty group for {video_id}")
            result[video_id] = group
        else:
            result[video_id] = video_id.split("_", 1)[0]
    return result


def make_grouped_folds(
    video_ids: Sequence[str],
    group_map: Mapping[str, str],
    n_folds: int,
    seed: int,
    strata: Optional[Mapping[str, Sequence[float]]] = None,
) -> List[dict]:
    """Create deterministic participant-disjoint, multilabel-balanced folds."""

    by_group: Dict[str, List[str]] = {}
    for video_id in sorted(video_ids):
        if video_id not in group_map:
            raise KeyError(f"Missing participant/group for {video_id}")
        by_group.setdefault(str(group_map[video_id]), []).append(video_id)
    if len(by_group) < 3:
        raise ValueError(
            "At least three participant groups are required for train/val/test"
        )
    if n_folds < 3:
        raise ValueError("n_folds must be at least 3")
    n_folds = min(int(n_folds), len(by_group))

    rng = random.Random(seed)
    groups = list(by_group)
    rng.shuffle(groups)
    if strata:
        width = len(next(iter(strata.values())))
        group_vectors = {}
        for group, members in by_group.items():
            vector = np.zeros(width + 1, dtype=np.float64)
            vector[0] = len(members)
            for video_id in members:
                values = np.asarray(
                    strata.get(video_id, np.zeros(width)), dtype=np.float64
                )
                if values.shape != (width,):
                    raise ValueError(f"Inconsistent stratum width for {video_id}")
                vector[1:] += values
            group_vectors[group] = vector
        target = sum(group_vectors.values()) / n_folds
        scale = np.where(target > 0, target, 1.0)
        groups.sort(
            key=lambda group: float(np.linalg.norm(group_vectors[group] / scale)),
            reverse=True,
        )
    else:
        group_vectors = {
            group: np.asarray([len(members)], dtype=np.float64)
            for group, members in by_group.items()
        }
        target = sum(group_vectors.values()) / n_folds
        scale = np.where(target > 0, target, 1.0)
        groups.sort(key=lambda group: len(by_group[group]), reverse=True)

    fold_groups: List[List[str]] = [[] for _ in range(n_folds)]
    fold_loads = np.zeros((n_folds, len(target)), dtype=np.float64)
    for group in groups:
        scores = []
        for fold_index in range(n_folds):
            proposal = fold_loads.copy()
            proposal[fold_index] += group_vectors[group]
            normalized = proposal / scale
            scores.append(float(np.var(normalized, axis=0).mean()))
        target_fold = min(range(n_folds), key=lambda idx: (scores[idx], idx))
        fold_groups[target_fold].append(group)
        fold_loads[target_fold] += group_vectors[group]

    folds = []
    for test_index in range(n_folds):
        val_index = (test_index + 1) % n_folds
        test_groups = set(fold_groups[test_index])
        val_groups = set(fold_groups[val_index])
        train_groups = set(groups) - test_groups - val_groups
        split = {
            "fold": test_index,
            "train": sorted(v for g in train_groups for v in by_group[g]),
            "val": sorted(v for g in val_groups for v in by_group[g]),
            "test": sorted(v for g in test_groups for v in by_group[g]),
            "train_groups": sorted(train_groups),
            "val_groups": sorted(val_groups),
            "test_groups": sorted(test_groups),
        }
        assert not (set(split["train_groups"]) & set(split["val_groups"]))
        assert not (set(split["train_groups"]) & set(split["test_groups"]))
        assert not (set(split["val_groups"]) & set(split["test_groups"]))
        folds.append(split)
    return folds


def save_json(path: str, payload: object) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def transitions_to_dicts(transitions: Sequence[TransitionRecord]) -> List[dict]:
    return [asdict(item) for item in transitions]
