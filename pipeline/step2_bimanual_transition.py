"""Train and evaluate expert-conditioned bimanual transition models.

The command is intentionally configuration-driven.  Every ablation uses
the same data, split, observation model, optimizer and metrics; a variant may
only override the explicit switches in the configuration file.

Examples
--------
Validate the full experiment without training::

    python pipeline/step2_bimanual_transition.py dry-run \
        --config configs/hact_reassembly_a.json

Run one reproducible fold::

    python pipeline/step2_bimanual_transition.py run \
        --config configs/hact_reassembly_a.json \
        --variant hact --fold 0 --seed 0

Run or print the complete experiment matrix::

    python pipeline/step2_bimanual_transition.py matrix \
        --config configs/hact_reassembly_a.json --print-only
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import os
import random
import sys
import time
from dataclasses import dataclass, replace
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

# Required by deterministic CUDA matrix multiplications.  It must be set
# before torch initializes a cuBLAS handle; callers may still override it.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import torch
import torch.nn.functional as F

_PIPELINE_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_PIPELINE_DIR)
_UTILS_DIR = os.path.join(_ROOT, "utils")
for _path in (_PIPELINE_DIR, _UTILS_DIR):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from bimanual_transition_data import (  # noqa: E402
    ANOMALY_TYPES,
    HANDS,
    PHASE_STATES,
    AnnotatedEvent,
    FactorVocabulary,
    VideoExample,
    annotation_path,
    audit_annotation_vocabulary,
    build_frame_targets,
    build_transitions,
    discover_video_ids,
    encode_sop_marks,
    encode_sop_requirements,
    feature_path,
    load_events,
    load_group_map,
    load_video_example,
    make_grouped_folds,
    save_json,
)
from bimanual_transition_model import (  # noqa: E402
    BimanualTransitionModel,
    LongContextObservationModel,
    TransitionBatch,
    decode_observed_events,
    make_transition_batch,
    observation_loss,
)
from bimanual_transition_metrics import (  # noqa: E402
    aggregate_transition_predictions,
    average_precision,
    evaluate_event_predictions,
    hierarchical_probabilities,
    hierarchical_decision,
    select_f1_threshold,
    softmax,
)
from procedure_scoring import evaluate_procedure_scores, score_procedure  # noqa: E402


def _deep_merge(base: Mapping[str, object], override: Mapping[str, object]) -> dict:
    merged = copy.deepcopy(dict(base))
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(merged.get(key), Mapping):
            merged[key] = _deep_merge(merged[key], value)  # type: ignore[arg-type]
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def _resolve_path(value: Optional[str], root: str) -> Optional[str]:
    if not value:
        return value
    expanded = os.path.expandvars(os.path.expanduser(str(value)))
    return os.path.abspath(
        expanded if os.path.isabs(expanded) else os.path.join(root, expanded)
    )


def load_config(path: str, variant: Optional[str] = None) -> dict:
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    root = os.path.dirname(os.path.dirname(os.path.abspath(path)))
    config = {
        key: copy.deepcopy(value) for key, value in raw.items() if key != "variants"
    }
    variants = raw.get("variants", {})
    if variant is not None:
        if variant not in variants:
            raise KeyError(
                f"Unknown variant {variant!r}; choose from {sorted(variants)}"
            )
        config = _deep_merge(config, variants[variant])
        config["variant"] = variant
    config["config_path"] = os.path.abspath(path)
    config["available_variants"] = sorted(variants)
    for key in (
        "features_dir",
        "annotations_dir",
        "expert_features_dir",
        "expert_annotations_dir",
        "sop_path",
        "groups_json",
        "verbs_path",
        "nouns_path",
        "ontology_path",
        "objects_path",
    ):
        if key in config.get("data", {}):
            config["data"][key] = _resolve_path(config["data"].get(key), root)
    config["output_root"] = _resolve_path(
        config.get("output_root", "outputs/bimanual_transition"), root
    )
    return config


def set_reproducible_seed(seed: int, deterministic_algorithms: bool = False) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(bool(deterministic_algorithms))


def resolve_device(requested: str) -> torch.device:
    if requested.startswith("cuda") and not torch.cuda.is_available():
        print("[warn] CUDA requested but unavailable; using CPU")
        return torch.device("cpu")
    return torch.device(requested)


def _finite_or_none(value: float) -> Optional[float]:
    return float(value) if math.isfinite(float(value)) else None


@dataclass
class EncodedVideo:
    video_id: str
    fps: float
    frame_ids: np.ndarray
    events: List[AnnotatedEvent]
    hidden: torch.Tensor
    logits: Dict[str, Dict[str, np.ndarray]]


def _target_tensors(
    example: VideoExample, device: torch.device
) -> Dict[str, torch.Tensor]:
    return {
        "verb": torch.as_tensor(example.targets.verb, dtype=torch.long, device=device),
        "part": torch.as_tensor(example.targets.part, dtype=torch.long, device=device),
        "tool": torch.as_tensor(example.targets.tool, dtype=torch.long, device=device),
        "phase": torch.as_tensor(
            example.targets.phase, dtype=torch.long, device=device
        ),
    }


def observation_class_counts(
    examples: Sequence[VideoExample], vocab: FactorVocabulary
) -> Dict[str, torch.Tensor]:
    sizes = {
        "verb": vocab.size("verb"),
        "part": vocab.size("part"),
        "tool": vocab.size("tool"),
        "phase": len(PHASE_STATES),
    }
    counts = {name: np.zeros(size, dtype=np.int64) for name, size in sizes.items()}
    for example in examples:
        for factor in counts:
            counts[factor] += np.bincount(
                getattr(example.targets, factor).reshape(-1),
                minlength=len(counts[factor]),
            )
    return {name: torch.from_numpy(value) for name, value in counts.items()}


def _mean_observation_loss(
    model: LongContextObservationModel,
    examples: Sequence[VideoExample],
    counts: Mapping[str, torch.Tensor],
    device: torch.device,
) -> float:
    if not examples:
        return float("nan")
    model.eval()
    losses = []
    with torch.no_grad():
        for example in examples:
            features = torch.as_tensor(
                example.features, dtype=torch.float32, device=device
            )
            _, logits = model(features)
            losses.append(
                float(
                    observation_loss(logits, _target_tensors(example, device), counts)
                )
            )
    return float(np.mean(losses))


def train_observation_model(
    model: LongContextObservationModel,
    train_examples: Sequence[VideoExample],
    val_examples: Sequence[VideoExample],
    training: Mapping[str, object],
    device: torch.device,
    seed: int,
) -> List[dict]:
    epochs = int(training.get("observation_epochs", 60))
    patience = int(training.get("observation_patience", 10))
    feature_dropout = float(training.get("feature_dropout", 0.0))
    feature_noise_std = float(training.get("feature_noise_std", 0.0))
    temporal_mask_probability = float(
        training.get("temporal_mask_probability", 0.0)
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training.get("observation_lr", 1e-3)),
        weight_decay=float(training.get("weight_decay", 1e-4)),
    )
    counts = observation_class_counts(train_examples, model.vocab)
    counts = {key: value.to(device) for key, value in counts.items()}
    generator = random.Random(seed)
    best_state = copy.deepcopy(model.state_dict())
    best_value = float("inf")
    stale = 0
    history = []

    for epoch in range(epochs):
        model.train()
        order = list(range(len(train_examples)))
        generator.shuffle(order)
        losses = []
        for index in order:
            example = train_examples[index]
            features = torch.as_tensor(
                example.features, dtype=torch.float32, device=device
            ).clone()
            if feature_dropout > 0.0:
                features = F.dropout(features, p=feature_dropout, training=True)
            if feature_noise_std > 0.0:
                features = features + feature_noise_std * torch.randn_like(features)
            if temporal_mask_probability > 0.0:
                temporal_mask = torch.rand(
                    len(features), device=features.device
                ) < temporal_mask_probability
                features[temporal_mask] = 0.0
            targets = _target_tensors(example, device)
            optimizer.zero_grad(set_to_none=True)
            _, logits = model(features)
            loss = observation_loss(logits, targets, counts)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach()))

        validation = _mean_observation_loss(model, val_examples, counts, device)
        selection = validation if math.isfinite(validation) else float(np.mean(losses))
        history.append(
            {
                "epoch": epoch + 1,
                "train_loss": float(np.mean(losses)),
                "val_loss": validation,
            }
        )
        print(
            f"  observation epoch {epoch + 1:03d}: "
            f"train={np.mean(losses):.4f} val={validation:.4f}"
        )
        if selection < best_value - 1e-6:
            best_value = selection
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                break
    model.load_state_dict(best_state)
    model.eval()
    return history


def encode_example(
    model: LongContextObservationModel,
    example: VideoExample,
    device: torch.device,
) -> EncodedVideo:
    model.eval()
    with torch.no_grad():
        features = torch.as_tensor(example.features, dtype=torch.float32, device=device)
        hidden, logits = model(features)
    numpy_logits = {
        hand: {
            factor: values.detach().cpu().numpy()
            for factor, values in hand_logits.items()
        }
        for hand, hand_logits in logits.items()
    }
    return EncodedVideo(
        video_id=example.video_id,
        fps=example.fps,
        frame_ids=example.frame_ids,
        events=example.events,
        hidden=hidden.detach().cpu(),
        logits=numpy_logits,
    )


def encode_unannotated_expert(
    model: LongContextObservationModel,
    video_id: str,
    features_dir: str,
    vocab: FactorVocabulary,
    device: torch.device,
    fps: float,
) -> EncodedVideo:
    with np.load(feature_path(features_dir, video_id)) as data:
        features = np.asarray(data["features"], dtype=np.float32)
        frame_ids = np.asarray(data["frame_ids"], dtype=np.int64)
    empty = build_frame_targets([], frame_ids)
    example = VideoExample(video_id, fps, features, frame_ids, [], empty)
    encoded = encode_example(model, example, device)
    encoded.events = decode_observed_events(encoded.frame_ids, encoded.logits, vocab)
    return encoded


def _supported_macro_f1(
    targets: Sequence[np.ndarray],
    predictions: Sequence[np.ndarray],
    exclude_background: bool,
) -> Optional[float]:
    if not targets:
        return None
    target = np.concatenate(targets)
    prediction = np.concatenate(predictions)
    classes = sorted(set(int(value) for value in target))
    if exclude_background:
        classes = [value for value in classes if value != 0]
    if not classes:
        return None
    scores = []
    for class_id in classes:
        true_positive = int(np.sum((target == class_id) & (prediction == class_id)))
        false_positive = int(np.sum((target != class_id) & (prediction == class_id)))
        false_negative = int(np.sum((target == class_id) & (prediction != class_id)))
        denominator = 2 * true_positive + false_positive + false_negative
        scores.append(2 * true_positive / denominator if denominator else 0.0)
    return float(np.mean(scores))


def observation_metrics(
    encoded: Sequence[EncodedVideo],
    examples: Mapping[str, VideoExample],
    vocab: FactorVocabulary,
) -> dict:
    correct = {
        hand: {factor: 0 for factor in LongContextObservationModel.FACTORS}
        for hand in HANDS
    }
    total = {
        hand: {factor: 0 for factor in LongContextObservationModel.FACTORS}
        for hand in HANDS
    }
    all_targets = {
        hand: {factor: [] for factor in LongContextObservationModel.FACTORS}
        for hand in HANDS
    }
    all_predictions = {
        hand: {factor: [] for factor in LongContextObservationModel.FACTORS}
        for hand in HANDS
    }
    offsets = {"start": [], "onset": [], "end": []}
    overlap_thresholds = (0.10, 0.25, 0.50)
    detection_counts = {
        threshold: {"tp": 0, "fp": 0, "fn": 0}
        for threshold in overlap_thresholds
    }
    for item in encoded:
        example = examples[item.video_id]
        for hand_index, hand in enumerate(HANDS):
            for factor in LongContextObservationModel.FACTORS:
                prediction = np.argmax(item.logits[hand][factor], axis=1)
                target = getattr(example.targets, factor)[hand_index]
                correct[hand][factor] += int(np.sum(prediction == target))
                total[hand][factor] += len(target)
                all_targets[hand][factor].append(target)
                all_predictions[hand][factor].append(prediction)
        predicted_events = decode_observed_events(item.frame_ids, item.logits, vocab)
        pairs = []
        for overlap_threshold in overlap_thresholds:
            matched, unmatched_predictions, unmatched_targets = match_events(
                predicted_events, example.events, minimum_iou=overlap_threshold
            )
            counts = detection_counts[overlap_threshold]
            counts["tp"] += len(matched)
            counts["fp"] += len(unmatched_predictions)
            counts["fn"] += len(unmatched_targets)
            if overlap_threshold == 0.10:
                pairs = matched
        for predicted, target in pairs:
            offsets["start"].append(abs(predicted.start_frame - target.start_frame))
            offsets["onset"].append(abs(predicted.onset_frame - target.onset_frame))
            offsets["end"].append(abs(predicted.end_frame - target.end_frame))
    return {
        "accuracy": {
            hand: {
                factor: correct[hand][factor] / max(total[hand][factor], 1)
                for factor in LongContextObservationModel.FACTORS
            }
            for hand in HANDS
        },
        "macro_f1_supported": {
            hand: {
                factor: _supported_macro_f1(
                    all_targets[hand][factor],
                    all_predictions[hand][factor],
                    exclude_background=False,
                )
                for factor in LongContextObservationModel.FACTORS
            }
            for hand in HANDS
        },
        "foreground_macro_f1_supported": {
            hand: {
                factor: _supported_macro_f1(
                    all_targets[hand][factor],
                    all_predictions[hand][factor],
                    exclude_background=True,
                )
                for factor in LongContextObservationModel.FACTORS
            }
            for hand in HANDS
        },
        "boundary_mae_frames": {
            name: float(np.mean(values)) if values else None
            for name, values in offsets.items()
        },
        "matched_events": len(offsets["onset"]),
        "event_detection": {
            key: value
            for overlap_threshold in overlap_thresholds
            for key, value in _event_detection_metrics(
                detection_counts[overlap_threshold], overlap_threshold
            ).items()
        },
    }


def _event_detection_metrics(counts: Mapping[str, int], threshold: float) -> dict:
    tp, fp, fn = counts["tp"], counts["fp"], counts["fn"]
    suffix = str(int(round(100 * threshold)))
    return {
        f"precision_at_{suffix}": tp / (tp + fp) if tp + fp else 0.0,
        f"recall_at_{suffix}": tp / (tp + fn) if tp + fn else 0.0,
        f"f1_at_{suffix}": (
            2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0
        ),
    }


def _temporal_iou(first: AnnotatedEvent, second: AnnotatedEvent) -> float:
    intersection = max(
        0,
        min(first.end_frame, second.end_frame)
        - max(first.start_frame, second.start_frame)
        + 1,
    )
    union = (
        max(first.end_frame, second.end_frame)
        - min(first.start_frame, second.start_frame)
        + 1
    )
    return intersection / union if union else 0.0


def match_events(
    predicted: Sequence[AnnotatedEvent],
    targets: Sequence[AnnotatedEvent],
    minimum_iou: float,
) -> Tuple[
    List[Tuple[AnnotatedEvent, AnnotatedEvent]],
    List[AnnotatedEvent],
    List[AnnotatedEvent],
]:
    candidates = []
    for predicted_index, prediction in enumerate(predicted):
        for target_index, target in enumerate(targets):
            if prediction.hand != target.hand:
                continue
            candidates.append(
                (_temporal_iou(prediction, target), predicted_index, target_index)
            )
    used_predicted, used_targets, pairs = set(), set(), []
    for overlap, predicted_index, target_index in sorted(candidates, reverse=True):
        if overlap < minimum_iou:
            break
        if predicted_index in used_predicted or target_index in used_targets:
            continue
        used_predicted.add(predicted_index)
        used_targets.add(target_index)
        pairs.append((predicted[predicted_index], targets[target_index]))
    unmatched_predicted = [
        item for index, item in enumerate(predicted) if index not in used_predicted
    ]
    unmatched_targets = [
        item for index, item in enumerate(targets) if index not in used_targets
    ]
    return pairs, unmatched_predicted, unmatched_targets


def _transition_batch(
    encoded: EncodedVideo,
    events: Sequence[AnnotatedEvent],
    use_phase_transitions: bool,
    device: torch.device,
) -> TransitionBatch:
    records = build_transitions(
        encoded.video_id, encoded.fps, encoded.frame_ids, events, use_phase_transitions
    )
    return make_transition_batch(records, encoded.hidden, device)


def _event_batches(
    videos: Sequence[EncodedVideo],
    use_phase_transitions: bool,
    device: torch.device,
) -> Dict[str, TransitionBatch]:
    batches = {}
    for video in videos:
        if not video.events:
            continue
        batches[video.video_id] = _transition_batch(
            video, video.events, use_phase_transitions, device
        )
    return batches


# Inverse actions of the assembly domain, used to synthesize restoration
# events in paired counterfactuals.  The relation is structural domain
# knowledge on the verb inventory, like the SOP; it is not fitted to data.
_INVERSE_VERB_NAMES: Dict[str, str] = {
    "attach": "remove",
    "insert": "remove",
    "seat": "remove",
    "remove": "attach",
    "tighten": "loosen",
    "loosen": "tighten",
    "hand_tighten": "hand_loosen",
    "hand_loosen": "hand_tighten",
    "pick_up": "place",
    "place": "pick_up",
    "store": "pick_up",
}


def _ontology_mark_maps(
    config: Mapping[str, object], vocab: FactorVocabulary
) -> Tuple[Dict[int, Dict[str, List[int]]], Dict[int, int]]:
    """Legal part/tool indices per verb, and the inverse-verb relation."""

    legal: Dict[int, Dict[str, List[int]]] = {}
    path = str(config.get("data", {}).get("ontology_path") or "")
    if path and os.path.exists(path):
        with open(path, encoding="utf-8-sig") as handle:
            for row in csv.DictReader(handle):
                if str(row.get("allowed", "1")).strip() != "1":
                    continue
                verb = str(row.get("verb", "")).strip()
                noun = str(row.get("noun", "")).strip()
                if not vocab.contains("verb", verb):
                    continue
                verb_index = vocab.encode("verb", verb)
                for factor in ("part", "tool"):
                    if vocab.contains(factor, noun):
                        members = legal.setdefault(verb_index, {}).setdefault(
                            factor, []
                        )
                        index = vocab.encode(factor, noun)
                        if index not in members:
                            members.append(index)
    inverse = {
        vocab.encode("verb", source): vocab.encode("verb", target)
        for source, target in _INVERSE_VERB_NAMES.items()
        if vocab.contains("verb", source) and vocab.contains("verb", target)
    }
    return legal, inverse


def _synthetic_negative_batches(
    sources: Sequence[Tuple[EncodedVideo, Sequence[AnnotatedEvent], str]],
    vocab: FactorVocabulary,
    use_phase_transitions: bool,
    device: torch.device,
    seed: int,
    mode: str = "marks",
    mark_maps: Optional[
        Tuple[Dict[int, Dict[str, List[int]]], Dict[int, int]]
    ] = None,
) -> Dict[str, TransitionBatch]:
    """Counterfactual procedural negatives for the anomaly stage.

    Each training video contributes one counterfactual copy per symbolic
    anomaly class: one normal event is rewritten in mark space (a different
    part, a different tool, or a same-hand order swap of two events'
    semantics), the rewritten events carry the corresponding anomaly label,
    and every derived quantity is rebuilt by the standard transition
    constructor.  Frames and visual features are untouched, and the
    construction has no magnitude constant: replacements are uniform over the
    remaining vocabulary and the swap partner is uniform over the same hand's
    normal events.
    """

    wrong_part = ANOMALY_TYPES.index("error_wrong_part")
    wrong_tool = ANOMALY_TYPES.index("error_wrong_tool")
    procedural = ANOMALY_TYPES.index("error_procedural")
    batches: Dict[str, TransitionBatch] = {}
    for video, base_events, view_tag in sources:
        original_events = list(base_events)
        eligible = [
            index
            for index, event in enumerate(original_events)
            if event.anomaly == 0 and not event.recovery
        ]
        if not eligible:
            continue
        digest = int.from_bytes(
            hashlib.sha256(video.video_id.encode("utf-8")).digest()[:4], "big"
        )
        seed_view = [] if view_tag == "oracle" else [1]
        key_suffix = "" if view_tag == "oracle" else f"::{view_tag}"
        for class_name, class_index in (
            ("wrong_part", wrong_part),
            ("wrong_tool", wrong_tool),
            ("procedural", procedural),
        ):
            rng = np.random.default_rng([int(seed), digest, class_index] + seed_view)
            events = list(original_events)
            if class_name in {"wrong_part", "wrong_tool"}:
                factor = "part" if class_name == "wrong_part" else "tool"
                size = vocab.size(factor)
                if size < 2:
                    continue
                target = int(rng.choice(eligible))
                original = getattr(events[target], factor)
                candidates: List[int] = []
                if mode == "paired" and mark_maps is not None:
                    legal_map = mark_maps[0]
                    candidates = [
                        index
                        for index in legal_map.get(events[target].verb, {}).get(
                            factor, []
                        )
                        if index != original
                    ]
                if candidates:
                    replacement = int(
                        candidates[int(rng.integers(len(candidates)))]
                    )
                else:
                    replacement = int(rng.integers(size - 1))
                    if replacement >= original:
                        replacement += 1
                events[target] = replace(
                    events[target],
                    **{factor: replacement},
                    anomaly=class_index,
                    event_id=f"{events[target].event_id}::synthetic_{class_name}",
                )
                if mode == "paired" and mark_maps is not None:
                    inverse_map = mark_maps[1]
                    inverse_verb = inverse_map.get(events[target].verb)
                    following = [
                        index
                        for index in eligible
                        if index != target
                        and events[index].hand == events[target].hand
                        and events[index].start_frame
                        > events[target].start_frame
                    ]
                    if inverse_verb is not None and following:
                        partner = min(
                            following,
                            key=lambda index: events[index].start_frame,
                        )
                        events[partner] = replace(
                            events[partner],
                            verb=inverse_verb,
                            part=events[target].part,
                            tool=events[target].tool,
                            anomaly=0,
                            recovery=True,
                            event_id=(
                                f"{original_events[partner].event_id}"
                                "::synthetic_recovery"
                            ),
                        )
            else:
                by_hand = {}
                for index in eligible:
                    by_hand.setdefault(events[index].hand, []).append(index)
                pairs = [
                    (first, second)
                    for members in by_hand.values()
                    for position, first in enumerate(members)
                    for second in members[position + 1 :]
                    if (
                        events[first].verb,
                        events[first].part,
                        events[first].tool,
                    )
                    != (
                        events[second].verb,
                        events[second].part,
                        events[second].tool,
                    )
                ]
                if not pairs:
                    continue
                first, second = pairs[int(rng.integers(len(pairs)))]
                for source, other in ((first, second), (second, first)):
                    events[source] = replace(
                        events[source],
                        verb=original_events[other].verb,
                        part=original_events[other].part,
                        tool=original_events[other].tool,
                        anomaly=class_index,
                        event_id=(
                            f"{original_events[source].event_id}"
                            f"::synthetic_{class_name}"
                        ),
                    )
            batches[
                f"{video.video_id}::synthetic_{class_name}{key_suffix}"
            ] = _transition_batch(
                video, events, use_phase_transitions, device
            )
    return batches


def _observation_corrupted_events(
    video: EncodedVideo, minimum_iou: float, vocab: FactorVocabulary
) -> Tuple[List[AnnotatedEvent], List[AnnotatedEvent], dict]:
    predicted = decode_observed_events(video.frame_ids, video.logits, vocab)
    pairs, unmatched_predictions, unmatched_targets = match_events(
        predicted, video.events, minimum_iou=minimum_iou
    )
    events = []
    offsets = {"start": [], "onset": [], "end": []}
    factor_errors = {"verb": 0, "part": 0, "tool": 0}
    for prediction, target in pairs:
        events.append(
            AnnotatedEvent(
                event_id=f"observed::{prediction.event_id}",
                hand=prediction.hand,
                start_frame=prediction.start_frame,
                onset_frame=prediction.onset_frame,
                end_frame=prediction.end_frame,
                verb=prediction.verb,
                part=prediction.part,
                tool=prediction.tool,
                anomaly=target.anomaly,
                recovery=target.recovery,
            )
        )
        offsets["start"].append(prediction.start_frame - target.start_frame)
        offsets["onset"].append(prediction.onset_frame - target.onset_frame)
        offsets["end"].append(prediction.end_frame - target.end_frame)
        for factor in factor_errors:
            factor_errors[factor] += int(
                getattr(prediction, factor) != getattr(target, factor)
            )
    events.sort(
        key=lambda event: (
            event.start_frame,
            event.onset_frame,
            event.end_frame,
            event.hand,
            event.event_id,
        )
    )
    anomaly_view_events = list(events)
    for prediction in unmatched_predictions:
        anomaly_view_events.append(
            AnnotatedEvent(
                event_id=f"observed_fp::{prediction.event_id}",
                hand=prediction.hand,
                start_frame=prediction.start_frame,
                onset_frame=prediction.onset_frame,
                end_frame=prediction.end_frame,
                verb=prediction.verb,
                part=prediction.part,
                tool=prediction.tool,
                anomaly=0,
                recovery=False,
            )
        )
    anomaly_view_events.sort(
        key=lambda event: (
            event.start_frame,
            event.onset_frame,
            event.end_frame,
            event.hand,
            event.event_id,
        )
    )
    return events, anomaly_view_events, {
        "video_id": video.video_id,
        "ground_truth_events": len(video.events),
        "decoded_events": len(predicted),
        "matched_events": len(pairs),
        "unmatched_decoded_events": len(unmatched_predictions),
        "missed_ground_truth_events": len(unmatched_targets),
        "boundary_signed_offsets": offsets,
        "factor_errors": factor_errors,
    }


def _observed_event_batches(
    videos: Sequence[EncodedVideo],
    use_phase_transitions: bool,
    minimum_iou: float,
    device: torch.device,
    vocab: FactorVocabulary,
) -> Tuple[Dict[str, TransitionBatch], Dict[str, TransitionBatch], dict]:
    matched_batches = {}
    anomaly_batches = {}
    rows = []
    fallback_videos = []
    for video in videos:
        matched_events, anomaly_events, report = _observation_corrupted_events(
            video, minimum_iou, vocab
        )
        if not matched_events and video.events:
            matched_events = list(video.events)
            fallback_videos.append(video.video_id)
        if not anomaly_events and video.events:
            anomaly_events = list(video.events)
        if matched_events:
            matched_batches[video.video_id] = _transition_batch(
                video, matched_events, use_phase_transitions, device
            )
        if anomaly_events:
            anomaly_batches[video.video_id] = _transition_batch(
                video, anomaly_events, use_phase_transitions, device
            )
        rows.append(report)

    def flattened(name: str) -> List[float]:
        return [
            float(value)
            for row in rows
            for value in row["boundary_signed_offsets"][name]
        ]

    matched = sum(int(row["matched_events"]) for row in rows)
    report = {
        "videos": len(videos),
        "ground_truth_events": sum(int(row["ground_truth_events"]) for row in rows),
        "decoded_events": sum(int(row["decoded_events"]) for row in rows),
        "matched_events": matched,
        "fallback_to_oracle_videos": fallback_videos,
        "boundary_error": {
            name: {
                "signed_mean_frames": (
                    float(np.mean(flattened(name))) if flattened(name) else None
                ),
                "mae_frames": (
                    float(np.mean(np.abs(flattened(name)))) if flattened(name) else None
                ),
            }
            for name in ("start", "onset", "end")
        },
        "factor_error_rate": {
            factor: (
                sum(int(row["factor_errors"][factor]) for row in rows) / matched
                if matched
                else None
            )
            for factor in ("verb", "part", "tool")
        },
        "per_video": rows,
    }
    return matched_batches, anomaly_batches, report


def _two_view_training_batches(
    oracle: Mapping[str, TransitionBatch],
    observed: Mapping[str, TransitionBatch],
    enabled: bool,
) -> Dict[str, TransitionBatch]:
    if not enabled:
        return dict(oracle)
    combined = {f"{video_id}::oracle": batch for video_id, batch in oracle.items()}
    combined.update(
        {f"{video_id}::observed": batch for video_id, batch in observed.items()}
    )
    return combined


def _mean_normative_loss(
    model: BimanualTransitionModel,
    batches: Mapping[str, TransitionBatch],
    expert_batches: Sequence[TransitionBatch],
    sop_marks: Sequence[Mapping[str, int]],
) -> float:
    if not batches:
        return float("nan")
    model.eval()
    values = []
    with torch.no_grad():
        for batch in batches.values():
            outputs = model(batch, expert_batches, sop_marks)
            values.append(float(model.normative_loss(outputs, batch)))
    return float(np.mean(values))


def train_normative_model(
    model: BimanualTransitionModel,
    train_batches: Mapping[str, TransitionBatch],
    val_batches: Mapping[str, TransitionBatch],
    expert_batches: Sequence[TransitionBatch],
    sop_marks: Sequence[Mapping[str, int]],
    training: Mapping[str, object],
    seed: int,
) -> List[dict]:
    if (
        not model.use_hand_identity
        and not model.use_event_semantics
        and not model.use_time_likelihood
    ):
        print("  normative stage skipped: visual-only model has no normative targets")
        return []
    for parameter in model.anomaly_head.parameters():
        parameter.requires_grad = False
    parameters = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    optimizer = torch.optim.AdamW(
        parameters,
        lr=float(training.get("normative_lr", 3e-4)),
        weight_decay=float(training.get("weight_decay", 1e-4)),
    )
    epochs = int(training.get("normative_epochs", 60))
    patience = int(training.get("normative_patience", 10))
    generator = random.Random(seed + 101)
    best_state = copy.deepcopy(model.state_dict())
    best_value = float("inf")
    stale = 0
    history = []
    video_ids = list(train_batches)

    for epoch in range(epochs):
        model.train()
        generator.shuffle(video_ids)
        losses = []
        for video_id in video_ids:
            optimizer.zero_grad(set_to_none=True)
            batch = train_batches[video_id]
            outputs = model(batch, expert_batches, sop_marks)
            loss = model.normative_loss(outputs, batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters, 1.0)
            optimizer.step()
            losses.append(float(loss.detach()))
        validation = _mean_normative_loss(model, val_batches, expert_batches, sop_marks)
        selection = validation if math.isfinite(validation) else float(np.mean(losses))
        history.append(
            {
                "epoch": epoch + 1,
                "train_loss": float(np.mean(losses)),
                "val_loss": validation,
            }
        )
        print(
            f"  normative epoch {epoch + 1:03d}: "
            f"train={np.mean(losses):.4f} val={validation:.4f}"
        )
        if selection < best_value - 1e-6:
            best_value = selection
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                break
    model.load_state_dict(best_state)
    return history


def anomaly_batch_class_counts(
    batches: Mapping[str, TransitionBatch]
) -> torch.Tensor:
    # The final slot separates recovery from ordinary normal execution for
    # recovery-state variants; legacy binary variants simply ignore it.
    counts = torch.zeros(len(ANOMALY_TYPES) + 1, dtype=torch.float64)
    for batch in batches.values():
        labels = batch.anomaly.detach().cpu()
        weights = batch.event_weight.detach().cpu().to(torch.float64)
        counts[: len(ANOMALY_TYPES)].scatter_add_(0, labels, weights)
        counts[len(ANOMALY_TYPES)] += weights[
            batch.recovery.detach().cpu().to(torch.bool)
        ].sum()
    return counts


def collect_transition_rows(
    model: BimanualTransitionModel,
    batches: Mapping[str, TransitionBatch],
    expert_batches: Sequence[TransitionBatch],
    sop_marks: Sequence[Mapping[str, int]],
    memory: Optional[Mapping[str, Optional[torch.Tensor]]] = None,
) -> List[dict]:
    model.eval()
    rows = []
    with torch.no_grad():
        for batch in batches.values():
            outputs = model(batch, expert_batches, sop_marks, memory=memory)
            logits = outputs["anomaly_logits"].detach().cpu().numpy()
            binary_logits = outputs["binary_logits"].detach().cpu().numpy()
            recovery_logits = outputs.get("recovery_logits")
            if recovery_logits is not None:
                recovery_logits = recovery_logits.detach().cpu().numpy()
            type_logits = outputs["type_logits"].detach().cpu().numpy()
            evidence = outputs["evidence"].detach().cpu().numpy()
            coordination = outputs.get("coordination_evidence")
            if coordination is not None:
                coordination = coordination.detach().cpu().numpy()
            cross_available = outputs["cross_available"].detach().cpu().numpy()
            supported_types = model.supported_anomaly_types.detach().cpu().tolist()
            for index, record in enumerate(batch.records):
                rows.append(
                    {
                        "video_id": record.video_id,
                        "event_id": record.event_id,
                        "event_start_frame": record.event_start_frame,
                        "event_end_frame": record.event_end_frame,
                        "frame": record.frame,
                        "hand": record.hand,
                        "transition": record.transition,
                        "verb": record.verb,
                        "part": record.part,
                        "tool": record.tool,
                        "target": record.anomaly,
                        "recovery": record.recovery,
                        "logits": logits[index].tolist(),
                        "binary_logits": binary_logits[index].tolist(),
                        **(
                            {"recovery_logits": recovery_logits[index].tolist()}
                            if recovery_logits is not None
                            else {}
                        ),
                        "type_logits": type_logits[index].tolist(),
                        "supported_anomaly_types": supported_types,
                        "evidence": {
                            name: float(evidence[index, evidence_index])
                            for evidence_index, name in enumerate(model.EVIDENCE_NAMES)
                        },
                        "coordination_evidence": (
                            {
                                name: float(coordination[index, evidence_index])
                                for evidence_index, name in enumerate(
                                    model.COORDINATION_EVIDENCE_NAMES
                                )
                            }
                            if coordination is not None
                            else {}
                        ),
                        "cross_available": bool(cross_available[index]),
                    }
                )
    return rows


def _mean_anomaly_loss(
    model: BimanualTransitionModel,
    batches: Mapping[str, TransitionBatch],
    expert_batches: Sequence[TransitionBatch],
    sop_marks: Sequence[Mapping[str, int]],
    class_counts: torch.Tensor,
    recovery_only: bool = False,
) -> float:
    if not batches:
        return float("nan")
    model.eval()
    values = []
    with torch.no_grad():
        for batch in batches.values():
            outputs = model(batch, expert_batches, sop_marks)
            values.append(
                float(
                    model.anomaly_loss(
                        outputs, batch, class_counts, recovery_only=recovery_only
                    )
                )
            )
    return float(np.mean(values))


def _validation_frame_auprc(
    events: Sequence[Mapping[str, object]],
    frame_targets: Mapping[str, np.ndarray],
) -> float:
    """Rasterize already-predicted events and score validation frame AP."""

    events_by_video: Dict[str, List[Mapping[str, object]]] = {}
    for event in events:
        events_by_video.setdefault(str(event["video_id"]), []).append(event)
    targets, scores = [], []
    for video_id, target in frame_targets.items():
        score = np.zeros(len(target), dtype=np.float64)
        for event in events_by_video.get(video_id, []):
            start = max(0, int(event["start_frame"]))
            end = min(len(score), int(event["end_frame"]) + 1)
            if end > start:
                score[start:end] = np.maximum(
                    score[start:end], float(event["anomaly_score"])
                )
        targets.append(np.asarray(target, dtype=np.int64))
        scores.append(score)
    if not targets:
        return float("nan")
    return average_precision(np.concatenate(targets), np.concatenate(scores))


def train_anomaly_calibrator(
    model: BimanualTransitionModel,
    train_batches: Mapping[str, TransitionBatch],
    val_batches: Mapping[str, TransitionBatch],
    expert_batches: Sequence[TransitionBatch],
    sop_marks: Sequence[Mapping[str, int]],
    class_counts: torch.Tensor,
    training: Mapping[str, object],
    seed: int,
    correction_only: bool = False,
    recovery_only: bool = False,
    frame_targets: Optional[Mapping[str, np.ndarray]] = None,
) -> List[dict]:
    if correction_only and recovery_only:
        raise ValueError("Correction and recovery stages must be trained separately")
    selection_metric = str(
        training.get("anomaly_selection_metric", "event_auprc")
    ).strip().lower()
    if selection_metric not in {
        "event_auprc",
        "frame_auprc",
        "balanced_validation_loss",
    }:
        raise ValueError(
            "training.anomaly_selection_metric must be 'event_auprc', "
            "'frame_auprc', or 'balanced_validation_loss', got "
            f"{selection_metric!r}"
        )
    if selection_metric == "frame_auprc" and frame_targets is None:
        raise ValueError(
            "frame_targets are required when anomaly_selection_metric='frame_auprc'"
        )

    def selection_score(
        event_auprc: object, frame_auprc: float, validation_loss: float
    ) -> float:
        if selection_metric == "balanced_validation_loss":
            return -float(validation_loss)
        selected = frame_auprc if selection_metric == "frame_auprc" else event_auprc
        if selected is not None and math.isfinite(float(selected)):
            return float(selected)
        return -float(validation_loss)

    for parameter in model.parameters():
        parameter.requires_grad = False
    for name, module in model.anomaly_head.items():
        is_correction = name.startswith("state_")
        is_recovery = name.startswith("recovery_")
        train_module = (
            is_recovery
            if recovery_only
            else is_correction
            if correction_only
            else not is_correction and not is_recovery
        )
        if train_module:
            for parameter in module.parameters():
                parameter.requires_grad = True
    model.correction_enabled = bool(
        model.use_correction_state and (correction_only or recovery_only)
    )
    model.recovery_gate_enabled = bool(model.use_recovery_gate and recovery_only)
    if hasattr(model, "cross_anomaly_adapter") and not correction_only and not recovery_only:
        for parameter in model.cross_anomaly_adapter.parameters():
            parameter.requires_grad = True
    if model.train_visual_during_anomaly and not correction_only and not recovery_only:
        for module in (model.visual_projection, model.token_norm):
            for parameter in module.parameters():
                parameter.requires_grad = True
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(
        parameters,
        lr=float(training.get("anomaly_lr", 1e-3)),
        weight_decay=float(training.get("weight_decay", 1e-4)),
    )
    epochs = int(training.get("anomaly_epochs", 50))
    patience = int(training.get("anomaly_patience", 10))
    generator = random.Random(seed + 202)
    best_state = copy.deepcopy(model.state_dict())
    best_score = -float("inf")
    stale = 0
    history = []
    video_ids = list(train_batches)
    class_counts = class_counts.to(next(model.parameters()).device)

    if correction_only or recovery_only:
        # The zero residual is a valid candidate and exactly recovers the
        # frozen base HACT.  Fine-tuning is retained only if validation AUPRC
        # improves over that starting point.
        rows = collect_transition_rows(model, val_batches, expert_batches, sop_marks)
        events = aggregate_transition_predictions(rows, ANOMALY_TYPES)
        initial_metrics = evaluate_event_predictions(events, ANOMALY_TYPES)
        initial_auprc = initial_metrics.get("anomaly_auprc", float("nan"))
        initial_frame_auprc = (
            _validation_frame_auprc(events, frame_targets)
            if frame_targets is not None
            else float("nan")
        )
        initial_loss = _mean_anomaly_loss(
            model,
            val_batches,
            expert_batches,
            sop_marks,
            class_counts,
            recovery_only=recovery_only,
        )
        best_score = selection_score(
            initial_auprc, initial_frame_auprc, initial_loss
        )
        history.append(
            {
                "epoch": 0,
                "train_loss": None,
                "val_loss": initial_loss,
                "val_anomaly_auprc": (
                    _finite_or_none(float(initial_auprc))
                    if initial_auprc is not None
                    else None
                ),
                "val_frame_auprc": _finite_or_none(initial_frame_auprc),
                "selection_metric": selection_metric,
                "selection_score": _finite_or_none(best_score),
            }
        )

    for epoch in range(epochs):
        # The normative model is a fixed evidence generator during
        # calibration; keep its dropout disabled and train only the head.
        model.eval()
        if correction_only:
            for name, module in model.anomaly_head.items():
                if name.startswith("state_"):
                    module.train()
        elif recovery_only:
            model.anomaly_head["recovery_encoder"].train()
        else:
            model.anomaly_head.train()
        if hasattr(model, "cross_anomaly_adapter") and not correction_only:
            model.cross_anomaly_adapter.train()
        generator.shuffle(video_ids)
        losses = []
        for video_id in video_ids:
            optimizer.zero_grad(set_to_none=True)
            batch = train_batches[video_id]
            outputs = model(batch, expert_batches, sop_marks)
            loss = model.anomaly_loss(
                outputs, batch, class_counts, recovery_only=recovery_only
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters, 1.0)
            optimizer.step()
            losses.append(float(loss.detach()))

        validation_loss = _mean_anomaly_loss(
            model,
            val_batches,
            expert_batches,
            sop_marks,
            class_counts,
            recovery_only=recovery_only,
        )
        rows = collect_transition_rows(model, val_batches, expert_batches, sop_marks)
        events = aggregate_transition_predictions(rows, ANOMALY_TYPES)
        metrics = evaluate_event_predictions(events, ANOMALY_TYPES)
        auprc = metrics.get("anomaly_auprc", float("nan"))
        frame_auprc = (
            _validation_frame_auprc(events, frame_targets)
            if frame_targets is not None
            else float("nan")
        )
        score = selection_score(auprc, frame_auprc, validation_loss)
        history.append(
            {
                "epoch": epoch + 1,
                "train_loss": float(np.mean(losses)),
                "val_loss": validation_loss,
                "val_anomaly_auprc": (
                    _finite_or_none(float(auprc)) if auprc is not None else None
                ),
                "val_frame_auprc": _finite_or_none(frame_auprc),
                "selection_metric": selection_metric,
                "selection_score": _finite_or_none(score),
            }
        )
        print(
            f"  anomaly epoch {epoch + 1:03d}: train={np.mean(losses):.4f} "
            f"val={validation_loss:.4f} event-AP={float(auprc):.4f} "
            f"frame-AP={float(frame_auprc):.4f}"
        )
        if score > best_score + 1e-6:
            best_score = score
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                break
    model.load_state_dict(best_state)
    model.correction_enabled = model.use_correction_state
    model.recovery_gate_enabled = model.use_recovery_gate
    model.eval()
    return history


def _fit_temperature_arrays(logits: Sequence[object], targets: Sequence[int]) -> float:
    if len(logits) < 2 or len(set(int(value) for value in targets)) < 2:
        return 1.0
    logits_tensor = torch.tensor(list(logits), dtype=torch.float64)
    targets_tensor = torch.tensor(list(targets), dtype=torch.long)
    log_temperature = torch.zeros((), dtype=torch.float64, requires_grad=True)
    optimizer = torch.optim.LBFGS(
        [log_temperature], lr=0.1, max_iter=50, line_search_fn="strong_wolfe"
    )

    def closure():
        optimizer.zero_grad()
        temperature = log_temperature.exp().clamp(0.05, 10.0)
        loss = torch.nn.functional.cross_entropy(
            logits_tensor / temperature, targets_tensor
        )
        loss.backward()
        return loss

    try:
        optimizer.step(closure)
        return float(log_temperature.detach().exp().clamp(0.05, 10.0))
    except RuntimeError:
        return 1.0


def fit_temperature(events: Sequence[Mapping[str, object]]) -> object:
    """Fit binary and conditional-type temperatures on validation events."""

    hierarchical = [
        event for event in events if "binary_logits" in event and "type_logits" in event
    ]
    if hierarchical:
        state_targets = []
        for event in hierarchical:
            if len(event["binary_logits"]) == 3:
                state_targets.append(
                    1
                    if int(event["target"]) != 0
                    else 2
                    if bool(event.get("recovery", False))
                    else 0
                )
            else:
                state_targets.append(int(int(event["target"]) != 0))
        binary_temperature = _fit_temperature_arrays(
            [event["binary_logits"] for event in hierarchical],
            state_targets,
        )
        typed = []
        for event in hierarchical:
            target = int(event["target"])
            support = event.get("supported_anomaly_types", [])
            if target > 0 and target - 1 < len(support) and bool(support[target - 1]):
                typed.append(event)
        type_temperature = _fit_temperature_arrays(
            [event["type_logits"] for event in typed],
            [int(event["target"]) - 1 for event in typed],
        )
        return {"binary": binary_temperature, "type": type_temperature}

    usable = [event for event in events if "logits" in event]
    return _fit_temperature_arrays(
        [event["logits"] for event in usable],
        [int(event["target"]) for event in usable],
    )


def apply_temperature(
    events: Sequence[Mapping[str, object]], temperature: object
) -> List[dict]:
    calibrated = []
    for source in events:
        event = dict(source)
        if "binary_logits" in event and "type_logits" in event:
            if isinstance(temperature, Mapping):
                binary_temperature = float(temperature.get("binary", 1.0))
                type_temperature = float(temperature.get("type", 1.0))
            else:
                binary_temperature = type_temperature = float(temperature)
            probabilities = hierarchical_probabilities(
                np.asarray(event["binary_logits"]),
                np.asarray(event["type_logits"]),
                binary_temperature,
                type_temperature,
                event.get("supported_anomaly_types"),
            )
            event["probabilities"] = probabilities.tolist()
            state_probabilities = softmax(
                np.asarray(event["binary_logits"]), binary_temperature
            )
            if len(state_probabilities) == 3:
                event["state_probabilities"] = state_probabilities.tolist()
            event["anomaly_score"] = float(1.0 - probabilities[0])
            event["predicted_type"] = hierarchical_decision(probabilities, 0.5)
            event["predicted_type_name"] = ANOMALY_TYPES[event["predicted_type"]]
        elif "logits" in event:
            probabilities = softmax(np.asarray(event["logits"]), temperature)
            event["probabilities"] = probabilities.tolist()
            event["anomaly_score"] = float(1.0 - probabilities[0])
            event["predicted_type"] = hierarchical_decision(probabilities, 0.5)
            event["predicted_type_name"] = ANOMALY_TYPES[event["predicted_type"]]
        transition_predictions = []
        for source_point in event.get("transition_predictions", []):
            point = dict(source_point)
            if "binary_logits" in point and "type_logits" in point:
                if isinstance(temperature, Mapping):
                    binary_temperature = float(temperature.get("binary", 1.0))
                    type_temperature = float(temperature.get("type", 1.0))
                else:
                    binary_temperature = type_temperature = float(temperature)
                point_probabilities = hierarchical_probabilities(
                    np.asarray(point["binary_logits"]),
                    np.asarray(point["type_logits"]),
                    binary_temperature,
                    type_temperature,
                    point.get("supported_anomaly_types"),
                )
                state_probabilities = softmax(
                    np.asarray(point["binary_logits"]), binary_temperature
                )
                if len(state_probabilities) == 3:
                    point["state_probabilities"] = state_probabilities.tolist()
            else:
                point_probabilities = softmax(
                    np.asarray(point["logits"]), float(temperature)
                )
            point_type = hierarchical_decision(point_probabilities, 0.5)
            point["probabilities"] = point_probabilities.tolist()
            point["anomaly_score"] = float(1.0 - point_probabilities[0])
            point["predicted_type"] = point_type
            point["predicted_type_name"] = ANOMALY_TYPES[point_type]
            transition_predictions.append(point)
        if transition_predictions:
            event["transition_predictions"] = transition_predictions
        calibrated.append(event)
    return calibrated


def apply_event_decisions(events: Sequence[dict], threshold: float) -> None:
    for event in events:
        probabilities = np.asarray(event["probabilities"], dtype=np.float64)
        predicted_type = hierarchical_decision(probabilities, threshold)
        event["decision_threshold"] = float(threshold)
        event["is_anomaly"] = bool(predicted_type != 0)
        event["predicted_type"] = predicted_type
        event["predicted_type_name"] = ANOMALY_TYPES[predicted_type]


def attach_factor_names(events: Sequence[dict], vocab: FactorVocabulary) -> None:
    for event in events:
        for factor in ("verb", "part", "tool"):
            index = int(event.get(factor, -1))
            event[f"{factor}_name"] = (
                vocab.decode(factor, index)
                if 0 <= index < vocab.size(factor)
                else "<unobserved>"
            )


def score_predicted_events(
    model: BimanualTransitionModel,
    videos: Sequence[EncodedVideo],
    expert_batches: Sequence[TransitionBatch],
    sop_marks: Sequence[Mapping[str, int]],
    use_phase_transitions: bool,
    minimum_iou: float,
    device: torch.device,
    memory: Optional[Mapping[str, Optional[torch.Tensor]]] = None,
) -> List[dict]:
    evaluation_rows = []
    for video in videos:
        predicted = decode_observed_events(video.frame_ids, video.logits, model.vocab)
        scored_by_id = {}
        if predicted:
            batch = _transition_batch(video, predicted, use_phase_transitions, device)
            rows = collect_transition_rows(
                model, {video.video_id: batch}, expert_batches, sop_marks, memory=memory
            )
            scored = aggregate_transition_predictions(rows, ANOMALY_TYPES)
            scored_by_id = {str(event["event_id"]): event for event in scored}

        pairs, unmatched_predicted, unmatched_targets = match_events(
            predicted, video.events, minimum_iou=minimum_iou
        )
        for prediction, target in pairs:
            event = dict(scored_by_id[prediction.event_id])
            event.update(
                {
                    "event_id": target.event_id,
                    "target": target.anomaly,
                    "target_name": ANOMALY_TYPES[target.anomaly],
                    "recovery": target.recovery,
                    "matched_iou": _temporal_iou(prediction, target),
                    "detected": True,
                    "is_false_positive": False,
                }
            )
            evaluation_rows.append(event)
        for prediction in unmatched_predicted:
            event = dict(scored_by_id[prediction.event_id])
            event.update(
                {
                    "event_id": f"false_positive::{prediction.event_id}",
                    "target": 0,
                    "target_name": "normal",
                    "recovery": False,
                    "matched_iou": 0.0,
                    "detected": True,
                    "is_false_positive": True,
                }
            )
            evaluation_rows.append(event)
        for target in unmatched_targets:
            probabilities = np.zeros(len(ANOMALY_TYPES), dtype=np.float64)
            probabilities[0] = 1.0
            evaluation_rows.append(
                {
                    "video_id": video.video_id,
                    "event_id": target.event_id,
                    "hand": target.hand,
                    "verb": target.verb,
                    "part": target.part,
                    "tool": target.tool,
                    "target": target.anomaly,
                    "target_name": ANOMALY_TYPES[target.anomaly],
                    "recovery": target.recovery,
                    "start_frame": target.start_frame,
                    "end_frame": target.end_frame,
                    "probabilities": probabilities.tolist(),
                    "anomaly_score": 0.0,
                    "predicted_type": 0,
                    "predicted_type_name": "normal",
                    "matched_iou": 0.0,
                    "detected": False,
                    "is_false_positive": False,
                    "evidence": {},
                }
            )
    return evaluation_rows


def _expert_ids(config: Mapping[str, object]) -> List[str]:
    data = config["data"]
    requested = [str(value) for value in data.get("expert_ids", [])]
    if requested:
        return requested
    if not bool(data.get("auto_discover_experts", False)):
        return []
    expert_features_dir = str(data.get("expert_features_dir") or data["features_dir"])
    if os.path.isdir(expert_features_dir):
        return sorted(
            name[: -len("_features.npz")]
            for name in os.listdir(expert_features_dir)
            if name.endswith("_features.npz")
        )
    return []


def prepare_data(config: Mapping[str, object], seed: int) -> dict:
    data = config["data"]
    features_dir = str(data["features_dir"])
    annotations_dir = str(data["annotations_dir"])
    expert_ids = _expert_ids(config)
    discovered = discover_video_ids(features_dir, annotations_dir)
    worker_ids = [
        video_id for video_id in discovered if video_id not in set(expert_ids)
    ]
    if not worker_ids:
        raise ValueError(
            f"No worker videos with both feature and parsed-annotation files in "
            f"{features_dir} and {annotations_dir}"
        )

    expert_annotation_mode = str(data.get("expert_annotation_mode", "auto"))
    if expert_annotation_mode not in {"auto", "decoded_only"}:
        raise ValueError(
            "data.expert_annotation_mode must be 'auto' or 'decoded_only', "
            f"got {expert_annotation_mode!r}"
        )
    expert_annotations_dir = str(data.get("expert_annotations_dir") or annotations_dir)
    worker_annotation_paths = [
        annotation_path(annotations_dir, video_id) for video_id in worker_ids
    ]
    expert_annotation_paths = (
        []
        if expert_annotation_mode == "decoded_only"
        else [
            annotation_path(expert_annotations_dir, video_id)
            for video_id in expert_ids
            if os.path.exists(annotation_path(expert_annotations_dir, video_id))
        ]
    )
    vocab = FactorVocabulary.build_from_ontology(
        verbs_path=str(data.get("verbs_path") or ""),
        nouns_path=str(data.get("nouns_path") or ""),
        ontology_path=data.get("ontology_path"),
        objects_path=data.get("objects_path"),
        sop_path=data.get("sop_path"),
    )
    vocabulary_unknowns = audit_annotation_vocabulary(
        worker_annotation_paths + expert_annotation_paths, vocab
    )
    if bool(data.get("strict_vocabulary", True)) and any(
        vocabulary_unknowns[factor] for factor in vocabulary_unknowns
    ):
        raise ValueError(
            "Annotations contain labels outside the fixed task ontology. "
            "Update the ontology files, not the fold vocabulary: "
            f"{vocabulary_unknowns}"
        )
    examples = {
        video_id: load_video_example(video_id, features_dir, annotations_dir, vocab)
        for video_id in worker_ids
    }
    strata = {}
    for video_id, example in examples.items():
        counts = np.zeros(len(ANOMALY_TYPES) + 1, dtype=np.float64)
        for event in example.events:
            counts[event.anomaly] += 1.0
            counts[-1] += float(event.recovery)
        strata[video_id] = counts
    group_map = load_group_map(worker_ids, data.get("groups_json"))
    split_seed = int(config.get("split", {}).get("seed", seed))
    folds = make_grouped_folds(
        worker_ids,
        group_map,
        int(config.get("split", {}).get("folds", 5)),
        split_seed,
        strata=strata,
    )
    for fold in folds:
        fold["anomaly_counts"] = {}
        for partition in ("train", "val", "test"):
            counts = np.zeros(len(ANOMALY_TYPES), dtype=np.int64)
            recovery = 0
            for video_id in fold[partition]:
                for event in examples[video_id].events:
                    counts[event.anomaly] += 1
                    recovery += int(event.recovery)
            fold["anomaly_counts"][partition] = {
                **{
                    name: int(counts[index]) for index, name in enumerate(ANOMALY_TYPES)
                },
                "recovery": recovery,
            }
    dimensions = {example.features.shape[1] for example in examples.values()}
    if len(dimensions) != 1:
        raise ValueError(
            f"Worker feature dimensions do not agree: {sorted(dimensions)}"
        )
    return {
        "worker_ids": worker_ids,
        "expert_ids": expert_ids,
        "group_map": group_map,
        "folds": folds,
        "vocab": vocab,
        "vocabulary_unknowns": vocabulary_unknowns,
        "examples": examples,
        "input_dim": dimensions.pop(),
    }


def _load_and_encode_experts(
    config: Mapping[str, object],
    expert_ids: Sequence[str],
    vocab: FactorVocabulary,
    observation: LongContextObservationModel,
    device: torch.device,
) -> List[EncodedVideo]:
    data = config["data"]
    features_dir = str(data.get("expert_features_dir") or data["features_dir"])
    annotations_dir = str(data.get("expert_annotations_dir") or data["annotations_dir"])
    annotation_mode = str(data.get("expert_annotation_mode", "auto"))
    default_fps = float(data.get("expert_fps", 25.0))
    fps_by_id = {
        str(key): float(value)
        for key, value in data.get("expert_fps_by_id", {}).items()
    }
    encoded = []
    for video_id in expert_ids:
        feature_file = feature_path(features_dir, video_id)
        if not os.path.exists(feature_file):
            raise FileNotFoundError(f"Missing expert feature cache: {feature_file}")
        parsed = annotation_path(annotations_dir, video_id)
        if annotation_mode != "decoded_only" and os.path.exists(parsed):
            example = load_video_example(video_id, features_dir, annotations_dir, vocab)
            if any(event.is_anomaly or event.recovery for event in example.events):
                raise ValueError(
                    f"Expert reference {video_id} contains anomaly/recovery events"
                )
            item = encode_example(observation, example, device)
            print(f"  expert {video_id}: {len(item.events)} annotated actions")
        else:
            item = encode_unannotated_expert(
                observation,
                video_id,
                features_dir,
                vocab,
                device,
                fps_by_id.get(video_id, default_fps),
            )
            print(
                f"  expert {video_id}: {len(item.events)} automatically decoded actions"
            )
        if not item.events:
            raise ValueError(
                f"Expert reference {video_id} produced no transition events"
            )
        encoded.append(item)
    return encoded


def _make_observation_model(
    config: Mapping[str, object],
    input_dim: int,
    vocab: FactorVocabulary,
    device: torch.device,
) -> LongContextObservationModel:
    values = config.get("observation", {})
    return LongContextObservationModel(
        input_dim=input_dim,
        vocab=vocab,
        hidden_dim=int(values.get("hidden_dim", 256)),
        max_dilation=int(values.get("max_dilation", 512)),
        dropout=float(values.get("dropout", 0.10)),
    ).to(device)


def _make_transition_model(
    config: Mapping[str, object],
    visual_dim: int,
    vocab: FactorVocabulary,
    device: torch.device,
) -> BimanualTransitionModel:
    values = config.get("transition", {})
    return BimanualTransitionModel(
        visual_dim=visual_dim,
        vocab=vocab,
        model_dim=int(values.get("model_dim", 256)),
        num_layers=int(values.get("num_layers", 2)),
        num_heads=int(values.get("num_heads", 4)),
        dropout=float(values.get("dropout", 0.10)),
        use_history=bool(values.get("use_history", True)),
        use_bimanual_context=bool(values.get("use_bimanual_context", True)),
        use_expert_memory=bool(values.get("use_expert_memory", True)),
        use_sop_memory=bool(values.get("use_sop_memory", True)),
        use_time_likelihood=bool(values.get("use_time_likelihood", True)),
        timing_clock=str(values.get("timing_clock", "context")),
        use_hand_identity=bool(values.get("use_hand_identity", True)),
        use_event_semantics=bool(values.get("use_event_semantics", True)),
        use_role_fusion=bool(values.get("use_role_fusion", False)),
        role_fusion_interactions=bool(values.get("role_fusion_interactions", True)),
        gate_execution_state=bool(values.get("gate_execution_state", True)),
        use_recovery_state=bool(values.get("use_recovery_state", False)),
        factorized_recovery_state=bool(
            values.get("factorized_recovery_state", False)
        ),
        use_recovery_gate=bool(values.get("use_recovery_gate", False)),
        use_correction_state=bool(values.get("use_correction_state", False)),
        correction_state_mode=str(values.get("correction_state_mode", "joint")),
        correction_use_error_trace=bool(
            values.get("correction_use_error_trace", False)
        ),
        correction_use_state_belief=bool(
            values.get("correction_use_state_belief", False)
        ),
        anomaly_hidden_dim=int(
            values.get("anomaly_hidden_dim", values.get("model_dim", 256))
        ),
        subtype_loss_weight=float(values.get("subtype_loss_weight", 1.0)),
        minimum_subtype_count=float(values.get("minimum_subtype_count", 1.0)),
        recovery_weight=float(values.get("recovery_weight", 1.0)),
        hard_negative_weight=float(values.get("hard_negative_weight", 1.0)),
        memory_dropout=float(values.get("memory_dropout", 0.0)),
        cross_hand_fusion=str(values.get("cross_hand_fusion", "joint")),
        cross_hand_dropout=float(values.get("cross_hand_dropout", 0.0)),
        cross_gate_bias=float(values.get("cross_gate_bias", -2.0)),
        cross_auxiliary_weight=float(values.get("cross_auxiliary_weight", 0.25)),
        train_visual_during_anomaly=bool(
            values.get("train_visual_during_anomaly", False)
        ),
        use_effect_evidence=bool(values.get("use_effect_evidence", False)),
    ).to(device)


def _json_safe_metrics(metrics: Mapping[str, object]) -> dict:
    result = {}
    for key, value in metrics.items():
        if isinstance(value, Mapping):
            result[key] = _json_safe_metrics(value)
        elif isinstance(value, float) and not math.isfinite(value):
            result[key] = None
        elif isinstance(value, np.generic):
            result[key] = value.item()
        else:
            result[key] = value
    return result


def _ground_truth_score_events(example: VideoExample) -> List[dict]:
    return [
        {
            "event_id": event.event_id,
            "hand": event.hand,
            "start_frame": event.start_frame,
            "end_frame": event.end_frame,
            "verb": event.verb,
            "part": event.part,
            "tool": event.tool,
            "quality": float(event.anomaly == 0),
            "anomaly_score": float(event.anomaly != 0),
        }
        for event in example.events
    ]


def _procedure_score_rows(
    predicted_events: Sequence[Mapping[str, object]],
    examples: Sequence[VideoExample],
    requirements: Optional[Mapping[str, object]],
) -> List[dict]:
    if not requirements:
        return []
    by_video: Dict[str, List[Mapping[str, object]]] = {}
    for event in predicted_events:
        if bool(event.get("detected", True)):
            by_video.setdefault(str(event["video_id"]), []).append(event)
    rows = []
    for example in examples:
        predicted_assessment = score_procedure(
            by_video.get(example.video_id, []), requirements
        )
        reference_assessment = score_procedure(
            _ground_truth_score_events(example), requirements
        )
        rows.append(
            {
                "video_id": example.video_id,
                "predicted_score": float(predicted_assessment["procedure_score"]),
                "reference_score": float(reference_assessment["procedure_score"]),
                "absolute_error": abs(
                    float(predicted_assessment["procedure_score"])
                    - float(reference_assessment["procedure_score"])
                ),
                "predicted_assessment": predicted_assessment,
                "reference_assessment": reference_assessment,
            }
        )
    return rows


def _run_directory(
    config: Mapping[str, object], variant: str, seed: int, fold: int
) -> str:
    return os.path.join(
        str(config["output_root"]), variant, f"seed_{seed}", f"fold_{fold}"
    )


def _observation_cache_signature(
    config: Mapping[str, object],
    observation: LongContextObservationModel,
    split: Mapping[str, object],
) -> Tuple[dict, str]:
    data = config["data"]
    training = config.get("training", {})
    observation_training_keys = (
        "deterministic_algorithms",
        "observation_epochs",
        "observation_patience",
        "observation_lr",
        "weight_decay",
        "feature_dropout",
        "feature_noise_std",
        "temporal_mask_probability",
    )
    files = {}
    for video_id in sorted(set(split["train"]) | set(split["val"])):
        paths = (
            feature_path(str(data["features_dir"]), video_id),
            annotation_path(str(data["annotations_dir"]), video_id),
        )
        files[video_id] = [
            {
                "path": os.path.abspath(path),
                "size": os.path.getsize(path),
                "mtime_ns": os.stat(path).st_mtime_ns,
            }
            for path in paths
        ]
    signature = {
        "format_version": 2,
        "model": observation.checkpoint_config(),
        "training": {key: training.get(key) for key in observation_training_keys},
        "train": list(split["train"]),
        "val": list(split["val"]),
        "files": files,
    }
    digest = hashlib.sha256(
        json.dumps(signature, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()[:16]
    return signature, digest


def run_experiment(
    config: Mapping[str, object],
    variant: str,
    fold_index: int,
    seed: int,
    requested_device: str,
) -> dict:
    start_time = time.time()
    set_reproducible_seed(
        seed,
        bool(config.get("training", {}).get("deterministic_algorithms", False)),
    )
    device = resolve_device(requested_device)
    prepared = prepare_data(config, seed)
    folds = prepared["folds"]
    if not 0 <= fold_index < len(folds):
        raise IndexError(f"fold={fold_index} outside [0,{len(folds) - 1}]")
    split = folds[fold_index]
    vocab: FactorVocabulary = prepared["vocab"]
    examples: Dict[str, VideoExample] = prepared["examples"]
    run_dir = _run_directory(config, variant, seed, fold_index)
    os.makedirs(run_dir, exist_ok=True)
    save_json(os.path.join(run_dir, "effective_config.json"), config)
    save_json(os.path.join(run_dir, "split.json"), split)
    save_json(os.path.join(run_dir, "vocabulary.json"), vocab.to_dict())

    train_examples = [examples[video_id] for video_id in split["train"]]
    val_examples = [examples[video_id] for video_id in split["val"]]
    test_examples = [examples[video_id] for video_id in split["test"]]
    print(
        f"[{variant} seed={seed} fold={fold_index}] "
        f"train={len(train_examples)} val={len(val_examples)} test={len(test_examples)} "
        f"device={device}"
    )

    observation = _make_observation_model(config, prepared["input_dim"], vocab, device)
    observation_signature, observation_digest = _observation_cache_signature(
        config, observation, split
    )
    shared_observation_dir = os.path.join(
        str(config["output_root"]),
        "_shared_observation",
        f"seed_{seed}",
        f"fold_{fold_index}",
    )
    shared_observation_path = os.path.join(
        shared_observation_dir, f"model_{observation_digest}.pt"
    )
    print(
        f"[observation] RF={observation.receptive_field} frames, "
        f"hidden={observation.hidden_dim}"
    )
    if os.path.exists(shared_observation_path):
        shared = torch.load(shared_observation_path, map_location=device)
        if (
            shared.get("vocab") != vocab.to_dict()
            or shared.get("split") != split
            or shared.get("signature") != observation_signature
        ):
            raise ValueError(
                f"Shared observation checkpoint is incompatible with this vocabulary/split: "
                f"{shared_observation_path}"
            )
        observation.load_state_dict(shared["state_dict"])
        observation.eval()
        observation_history = shared.get("history", [])
        print(f"[observation] reused {shared_observation_path}")
    else:
        observation_history = train_observation_model(
            observation,
            train_examples,
            val_examples,
            config.get("training", {}),
            device,
            seed,
        )
        os.makedirs(shared_observation_dir, exist_ok=True)
        temporary = f"{shared_observation_path}.{os.getpid()}.tmp"
        torch.save(
            {
                "state_dict": observation.state_dict(),
                "signature": observation_signature,
                "vocab": vocab.to_dict(),
                "split": split,
                "history": observation_history,
            },
            temporary,
        )
        os.replace(temporary, shared_observation_path)
        print(f"[observation] cached {shared_observation_path}")
    save_json(os.path.join(run_dir, "observation_history.json"), observation_history)

    encoded_workers = {
        video_id: encode_example(observation, example, device)
        for video_id, example in examples.items()
    }
    observation_report = observation_metrics(
        [encoded_workers[video_id] for video_id in split["test"]], examples, vocab
    )
    save_json(os.path.join(run_dir, "observation_metrics.json"), observation_report)

    expert_ids = prepared["expert_ids"]
    transition_values = config.get("transition", {})
    needs_expert = bool(transition_values.get("use_expert_memory", True))
    if needs_expert and not expert_ids:
        raise ValueError(
            "This variant enables expert memory but data.expert_ids is empty and no "
            "expert feature caches were discovered"
        )
    encoded_experts = (
        _load_and_encode_experts(config, expert_ids, vocab, observation, device)
        if needs_expert and expert_ids
        else []
    )

    use_phase = bool(transition_values.get("use_phase_transitions", True))
    train_encoded = [encoded_workers[video_id] for video_id in split["train"]]
    val_encoded = [encoded_workers[video_id] for video_id in split["val"]]
    test_encoded = [encoded_workers[video_id] for video_id in split["test"]]
    minimum_iou = float(config.get("evaluation", {}).get("event_match_iou", 0.10))
    oracle_train_batches = _event_batches(train_encoded, use_phase, device)
    oracle_val_batches = _event_batches(val_encoded, use_phase, device)
    oracle_test_batches = _event_batches(test_encoded, use_phase, device)
    (
        observed_train_batches,
        observed_train_anomaly_batches,
        observed_train_report,
    ) = _observed_event_batches(train_encoded, use_phase, minimum_iou, device, vocab)
    (
        observed_val_batches,
        observed_val_anomaly_batches,
        observed_val_report,
    ) = _observed_event_batches(val_encoded, use_phase, minimum_iou, device, vocab)
    augmentation_enabled = bool(
        config.get("augmentation", {}).get("observed_event_view", True)
    )
    normative_train_batches = _two_view_training_batches(
        oracle_train_batches, observed_train_batches, augmentation_enabled
    )
    anomaly_train_batches = _two_view_training_batches(
        oracle_train_batches, observed_train_anomaly_batches, augmentation_enabled
    )
    synthetic_setting = config.get("training", {}).get(
        "synthetic_procedural_negatives", False
    )
    if isinstance(synthetic_setting, str):
        synthetic_mode = synthetic_setting.strip().lower()
        if synthetic_mode not in {"marks", "paired", "paired_two_view"}:
            raise ValueError(
                "training.synthetic_procedural_negatives must be a boolean, "
                "'marks', 'paired', or 'paired_two_view'"
            )
        synthetic_negatives_enabled = True
    else:
        synthetic_negatives_enabled = bool(synthetic_setting)
        synthetic_mode = "marks"
    synthetic_batches: Dict[str, TransitionBatch] = {}
    if synthetic_negatives_enabled:
        mark_maps = (
            _ontology_mark_maps(config, vocab)
            if synthetic_mode in {"paired", "paired_two_view"}
            else None
        )
        synthetic_sources = [
            (video, video.events, "oracle") for video in train_encoded
        ]
        if synthetic_mode == "paired_two_view":
            for video in train_encoded:
                _, decoded_events, _ = _observation_corrupted_events(
                    video, minimum_iou, vocab
                )
                if decoded_events:
                    synthetic_sources.append((video, decoded_events, "observed"))
        synthetic_batches = _synthetic_negative_batches(
            synthetic_sources,
            vocab,
            use_phase,
            device,
            seed,
            mode="paired" if synthetic_mode == "paired_two_view" else synthetic_mode,
            mark_maps=mark_maps,
        )
        anomaly_train_batches = {**anomaly_train_batches, **synthetic_batches}
    # Model selection always uses the deployment-like observed event view.
    normative_val_batches = observed_val_batches
    anomaly_val_batches = observed_val_anomaly_batches
    test_batches = oracle_test_batches
    save_json(
        os.path.join(run_dir, "event_noise_report.json"),
        {
            "augmentation_enabled": augmentation_enabled,
            "training_views": (
                ["oracle", "observation_corrupted"]
                if augmentation_enabled
                else ["oracle"]
            ),
            "selection_view": "observation_corrupted",
            "synthetic_procedural_negatives": len(synthetic_batches),
            "synthetic_negative_mode": (
                synthetic_mode if synthetic_negatives_enabled else None
            ),
            "train": observed_train_report,
            "validation": observed_val_report,
        },
    )
    expert_batches = list(_event_batches(encoded_experts, use_phase, device).values())
    sop_marks = encode_sop_marks(config["data"].get("sop_path"), vocab)
    sop_requirements = encode_sop_requirements(
        config["data"].get("sop_path"), vocab
    )

    transition_model = _make_transition_model(
        config, 2 * observation.hidden_dim, vocab, device
    )
    train_anomaly_counts = anomaly_batch_class_counts(anomaly_train_batches)
    validation_frame_targets = {}
    for example in val_examples:
        target = np.zeros(len(example.frame_ids), dtype=np.int64)
        for event in example.events:
            if event.anomaly == 0:
                continue
            start = max(0, int(event.start_frame))
            end = min(len(target), int(event.end_frame) + 1)
            if end > start:
                target[start:end] = 1
        validation_frame_targets[example.video_id] = target
    transition_model.set_anomaly_support(train_anomaly_counts)
    normative_history = train_normative_model(
        transition_model,
        normative_train_batches,
        normative_val_batches,
        expert_batches,
        sop_marks,
        config.get("training", {}),
        seed,
    )
    save_json(os.path.join(run_dir, "normative_history.json"), normative_history)
    anomaly_history = train_anomaly_calibrator(
        transition_model,
        anomaly_train_batches,
        anomaly_val_batches,
        expert_batches,
        sop_marks,
        train_anomaly_counts,
        config.get("training", {}),
        seed,
        frame_targets=validation_frame_targets,
    )
    save_json(os.path.join(run_dir, "anomaly_history.json"), anomaly_history)
    correction_history = []
    if transition_model.use_correction_state:
        correction_history = train_anomaly_calibrator(
            transition_model,
            anomaly_train_batches,
            anomaly_val_batches,
            expert_batches,
            sop_marks,
            train_anomaly_counts,
            config.get("training", {}),
            seed,
            correction_only=True,
            frame_targets=validation_frame_targets,
        )
        save_json(
            os.path.join(run_dir, "correction_history.json"), correction_history
        )
    recovery_history = []
    if transition_model.use_recovery_gate:
        recovery_history = train_anomaly_calibrator(
            transition_model,
            anomaly_train_batches,
            anomaly_val_batches,
            expert_batches,
            sop_marks,
            train_anomaly_counts,
            config.get("training", {}),
            seed,
            recovery_only=True,
            frame_targets=validation_frame_targets,
        )
        save_json(os.path.join(run_dir, "recovery_history.json"), recovery_history)

    transition_model.eval()
    with torch.no_grad():
        reference_memory = transition_model.build_memory(expert_batches, sop_marks)

    oracle_val_rows = collect_transition_rows(
        transition_model,
        oracle_val_batches,
        expert_batches,
        sop_marks,
        memory=reference_memory,
    )
    oracle_test_rows = collect_transition_rows(
        transition_model,
        test_batches,
        expert_batches,
        sop_marks,
        memory=reference_memory,
    )
    oracle_val = aggregate_transition_predictions(oracle_val_rows, ANOMALY_TYPES)
    oracle_test = aggregate_transition_predictions(oracle_test_rows, ANOMALY_TYPES)

    predicted_val = score_predicted_events(
        transition_model,
        val_encoded,
        expert_batches,
        sop_marks,
        use_phase,
        minimum_iou,
        device,
        memory=reference_memory,
    )
    predicted_test = score_predicted_events(
        transition_model,
        test_encoded,
        expert_batches,
        sop_marks,
        use_phase,
        minimum_iou,
        device,
        memory=reference_memory,
    )

    # Calibrate each evaluation protocol against its matching validation view.
    # Using predicted-event validation for oracle-event test scores mixes two
    # different score distributions and can depress thresholded metrics.
    oracle_temperature = fit_temperature(oracle_val)
    calibrated_oracle_val = apply_temperature(oracle_val, oracle_temperature)
    oracle_test = apply_temperature(oracle_test, oracle_temperature)
    oracle_threshold = select_f1_threshold(
        [int(event["target"] != 0) for event in calibrated_oracle_val],
        [float(event["anomaly_score"]) for event in calibrated_oracle_val],
    )

    predicted_source = predicted_val if predicted_val else oracle_val
    predicted_temperature = fit_temperature(predicted_source)
    predicted_val = apply_temperature(predicted_val, predicted_temperature)
    predicted_test = apply_temperature(predicted_test, predicted_temperature)
    calibrated_predicted_source = apply_temperature(
        predicted_source, predicted_temperature
    )
    predicted_threshold = select_f1_threshold(
        [int(event["target"] != 0) for event in calibrated_predicted_source],
        [float(event["anomaly_score"]) for event in calibrated_predicted_source],
    )
    oracle_val = calibrated_oracle_val
    for collection in (oracle_val, oracle_test):
        apply_event_decisions(collection, oracle_threshold)
        attach_factor_names(collection, vocab)
    for collection in (predicted_val, predicted_test):
        apply_event_decisions(collection, predicted_threshold)
        attach_factor_names(collection, vocab)

    procedure_rows = _procedure_score_rows(
        predicted_test, test_examples, sop_requirements
    )
    procedure_metrics = evaluate_procedure_scores(procedure_rows)

    metrics = {
        "variant": variant,
        "seed": seed,
        "fold": fold_index,
        # Backward-compatible deployment values use the predicted-event view.
        "temperature": predicted_temperature,
        "threshold": predicted_threshold,
        "calibration": {
            "oracle_event": {
                "temperature": oracle_temperature,
                "threshold": oracle_threshold,
            },
            "predicted_event": {
                "temperature": predicted_temperature,
                "threshold": predicted_threshold,
            },
        },
        "observation": observation_report,
        "validation_predicted_event": evaluate_event_predictions(
            predicted_val, ANOMALY_TYPES, predicted_threshold
        ),
        "validation_oracle_event": evaluate_event_predictions(
            oracle_val, ANOMALY_TYPES, oracle_threshold
        ),
        "oracle_event": evaluate_event_predictions(
            oracle_test, ANOMALY_TYPES, oracle_threshold
        ),
        "predicted_event": evaluate_event_predictions(
            predicted_test, ANOMALY_TYPES, predicted_threshold
        ),
        "procedure_score": procedure_metrics,
        "runtime_seconds": time.time() - start_time,
        "split_sizes": {key: len(split[key]) for key in ("train", "val", "test")},
    }
    metrics = _json_safe_metrics(metrics)
    save_json(os.path.join(run_dir, "metrics.json"), metrics)
    save_json(os.path.join(run_dir, "validation_event_predictions.json"), predicted_val)
    save_json(
        os.path.join(run_dir, "validation_oracle_event_predictions.json"), oracle_val
    )
    save_json(os.path.join(run_dir, "oracle_event_predictions.json"), oracle_test)
    save_json(os.path.join(run_dir, "predicted_event_predictions.json"), predicted_test)
    save_json(os.path.join(run_dir, "procedure_scores.json"), procedure_rows)

    torch.save(
        {
            "format_version": 2,
            "variant": variant,
            "seed": seed,
            "fold": fold_index,
            "vocab": vocab.to_dict(),
            "anomaly_types": list(ANOMALY_TYPES),
            "observation_config": observation.checkpoint_config(),
            "observation_state": observation.state_dict(),
            "transition_config": transition_model.checkpoint_config(),
            "transition_state": transition_model.state_dict(),
            "use_phase_transitions": use_phase,
            "temperature": predicted_temperature,
            "threshold": predicted_threshold,
            "oracle_temperature": oracle_temperature,
            "oracle_threshold": oracle_threshold,
            "expert_ids": expert_ids,
            "sop_marks": sop_marks,
            "sop_requirements": sop_requirements,
            "reference_memory": {
                key: value.detach().cpu() if value is not None else None
                for key, value in reference_memory.items()
            },
            "split": split,
        },
        os.path.join(run_dir, "model.pt"),
    )
    print(
        f"[done] {variant} seed={seed} fold={fold_index}: "
        f"AUPRC={metrics['predicted_event'].get('anomaly_auprc')} -> {run_dir}"
    )
    return metrics


def dry_run(config: Mapping[str, object], seed: int) -> dict:
    prepared = prepare_data(config, seed)
    examples = prepared["examples"]
    anomaly_counts = {name: 0 for name in ANOMALY_TYPES}
    recovery_count = 0
    overlap_count = 0
    for example in examples.values():
        overlap_count += example.targets.overlap_count
        for event in example.events:
            anomaly_counts[ANOMALY_TYPES[event.anomaly]] += 1
            recovery_count += int(event.recovery)
    warnings = []
    fold_count = len(prepared["folds"])
    groups_by_type = {name: set() for name in ANOMALY_TYPES}
    for video_id, example in examples.items():
        group = prepared["group_map"][video_id]
        for event in example.events:
            groups_by_type[ANOMALY_TYPES[event.anomaly]].add(group)
    for name, count in anomaly_counts.items():
        group_support = len(groups_by_type[name])
        if name != "normal" and group_support < fold_count:
            warnings.append(
                f"{name} has {count} event(s) from {group_support} participant group(s), "
                "so it cannot be represented in every test fold"
            )
    if not config["data"].get("groups_json") and len(
        set(prepared["group_map"].values())
    ) == len(prepared["worker_ids"]):
        warnings.append(
            "Every video inferred a different participant ID; provide data.groups_json "
            "if any participant has multiple videos"
        )
    expert_features_dir = str(
        config["data"].get("expert_features_dir") or config["data"]["features_dir"]
    )
    missing_experts = [
        video_id
        for video_id in prepared["expert_ids"]
        if not os.path.exists(feature_path(expert_features_dir, video_id))
    ]
    if missing_experts:
        warnings.append(f"Missing expert feature caches: {missing_experts}")
    if (
        config.get("transition", {}).get("use_expert_memory", True)
        and not prepared["expert_ids"]
    ):
        warnings.append("Expert memory is enabled but data.expert_ids is empty")
    if overlap_count:
        warnings.append(
            f"Found {overlap_count} same-hand overlap frame(s); inspect whether the "
            "one-active-event-per-hand annotation contract is valid"
        )

    report = {
        "worker_videos": len(prepared["worker_ids"]),
        "expert_ids": prepared["expert_ids"],
        "participant_groups": len(set(prepared["group_map"].values())),
        "feature_dim": prepared["input_dim"],
        "vocabulary_sizes": {
            factor: prepared["vocab"].size(factor)
            for factor in ("verb", "part", "tool")
        },
        "vocabulary_source": "fixed_task_ontology_and_sop",
        "vocabulary_unknowns": prepared["vocabulary_unknowns"],
        "anomaly_counts": anomaly_counts,
        "recovery_events": recovery_count,
        "same_hand_overlap_frames": overlap_count,
        "warnings": warnings,
        "folds": prepared["folds"],
    }
    print(json.dumps(report, indent=2))
    return report


def load_trained_models(
    checkpoint_path: str, device: torch.device
) -> Tuple[
    dict, FactorVocabulary, LongContextObservationModel, BimanualTransitionModel
]:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    vocab = FactorVocabulary.from_dict(checkpoint["vocab"])
    observation_config = dict(checkpoint["observation_config"])
    observation_config.pop("vocab", None)
    observation = LongContextObservationModel(
        vocab=vocab,
        input_dim=int(observation_config["input_dim"]),
        hidden_dim=int(observation_config["hidden_dim"]),
        max_dilation=int(observation_config["max_dilation"]),
        dropout=float(observation_config["dropout"]),
    ).to(device)
    observation.load_state_dict(checkpoint["observation_state"])
    observation.eval()

    transition_config = dict(checkpoint["transition_config"])
    transition_config.pop("vocab", None)
    transition = BimanualTransitionModel(vocab=vocab, **transition_config).to(device)
    transition.load_state_dict(checkpoint["transition_state"])
    transition.eval()
    return checkpoint, vocab, observation, transition


def predict_video(
    checkpoint_path: str,
    query_features_path: str,
    output_path: str,
    fps: float,
    requested_device: str,
    expert_features_dir: Optional[str] = None,
    expert_annotations_dir: Optional[str] = None,
    expert_ids: Optional[Sequence[str]] = None,
) -> dict:
    started = time.perf_counter()
    device = resolve_device(requested_device)
    checkpoint, vocab, observation, transition = load_trained_models(
        checkpoint_path, device
    )
    with np.load(query_features_path) as data:
        features = np.asarray(data["features"], dtype=np.float32)
        frame_ids = np.asarray(data["frame_ids"], dtype=np.int64)
    if features.shape[1] != observation.input_dim:
        raise ValueError(
            f"Feature dimension {features.shape[1]} does not match checkpoint "
            f"dimension {observation.input_dim}"
        )
    video_id = os.path.basename(query_features_path).replace("_features.npz", "")
    empty_targets = build_frame_targets([], frame_ids)
    query = encode_example(
        observation,
        VideoExample(video_id, float(fps), features, frame_ids, [], empty_targets),
        device,
    )
    query.events = decode_observed_events(frame_ids, query.logits, vocab)

    selected_experts = (
        list(expert_ids)
        if expert_ids is not None
        else list(checkpoint.get("expert_ids", []))
    )
    encoded_experts: List[EncodedVideo] = []
    stored_memory = checkpoint.get("reference_memory")
    reference_memory = None
    if stored_memory is not None:
        reference_memory = {
            key: value.to(device) if value is not None else None
            for key, value in stored_memory.items()
        }
    elif transition.use_expert_memory:
        if not selected_experts or not expert_features_dir:
            raise ValueError(
                "This checkpoint uses expert memory; provide --expert_features_dir "
                "and expert feature IDs"
            )
        expert_config = {
            "data": {
                "features_dir": expert_features_dir,
                "annotations_dir": expert_annotations_dir or "",
                "expert_features_dir": expert_features_dir,
                "expert_annotations_dir": expert_annotations_dir or "",
                "expert_fps": float(fps),
            }
        }
        encoded_experts = _load_and_encode_experts(
            expert_config, selected_experts, vocab, observation, device
        )

    use_phase = bool(checkpoint.get("use_phase_transitions", True))
    expert_batches = list(_event_batches(encoded_experts, use_phase, device).values())
    sop_marks = checkpoint.get("sop_marks", [])
    if reference_memory is None:
        with torch.no_grad():
            reference_memory = transition.build_memory(expert_batches, sop_marks)
    if query.events:
        batch = _transition_batch(query, query.events, use_phase, device)
        rows = collect_transition_rows(
            transition,
            {video_id: batch},
            expert_batches,
            sop_marks,
            memory=reference_memory,
        )
        events = aggregate_transition_predictions(
            rows, ANOMALY_TYPES, temperature=checkpoint.get("temperature", 1.0)
        )
    else:
        events = []
    threshold = float(checkpoint.get("threshold", 0.5))
    apply_event_decisions(events, threshold)
    attach_factor_names(events, vocab)
    procedure_assessment = score_procedure(
        events, checkpoint.get("sop_requirements")
    )
    report = {
        "video_id": video_id,
        "variant": checkpoint.get("variant"),
        "anomaly_types": list(ANOMALY_TYPES),
        "supported_anomaly_types": [
            ANOMALY_TYPES[index + 1]
            for index, supported in enumerate(
                transition.supported_anomaly_types.detach().cpu().tolist()
            )
            if supported
        ],
        "temperature": checkpoint.get("temperature", 1.0),
        "anomaly_threshold": threshold,
        "procedure_score": procedure_assessment["procedure_score"],
        "procedure_assessment": procedure_assessment,
        "n_detected_events": len(events),
        "model_runtime_seconds": time.perf_counter() - started,
        "events": events,
    }
    save_json(output_path, report)
    return report


def _override_expert_ids(config: dict, values: Optional[Sequence[str]]) -> None:
    if values is not None:
        config.setdefault("data", {})["expert_ids"] = list(values)


def _matrix_commands(
    config_path: str,
    variants: Sequence[str],
    folds: Sequence[int],
    seeds: Sequence[int],
    device: str,
) -> List[str]:
    commands = []
    for variant in variants:
        for seed in seeds:
            for fold in folds:
                commands.append(
                    "python pipeline/step2_bimanual_transition.py run "
                    f'--config "{config_path}" --variant {variant} --seed {seed} '
                    f"--fold {fold} --device {device}"
                )
    return commands


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Expert-conditioned bimanual phase-transition anomaly modelling"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    dry = subparsers.add_parser(
        "dry-run", help="Validate data, labels and participant folds"
    )
    dry.add_argument("--config", required=True)
    dry.add_argument("--variant", default="full")
    dry.add_argument("--seed", type=int, default=0)
    dry.add_argument("--expert_ids", nargs="*", default=None)

    run = subparsers.add_parser("run", help="Train and evaluate one variant/fold/seed")
    run.add_argument("--config", required=True)
    run.add_argument("--variant", required=True)
    run.add_argument("--fold", type=int, required=True)
    run.add_argument("--seed", type=int, default=0)
    run.add_argument("--device", default="cuda")
    run.add_argument("--expert_ids", nargs="*", default=None)

    matrix = subparsers.add_parser(
        "matrix", help="Print or execute the full ablation matrix"
    )
    matrix.add_argument("--config", required=True)
    matrix.add_argument("--variants", nargs="*", default=None)
    matrix.add_argument("--folds", type=int, nargs="*", default=None)
    matrix.add_argument("--seeds", type=int, nargs="*", default=None)
    matrix.add_argument("--device", default="cuda")
    matrix.add_argument("--print-only", action="store_true")
    matrix.add_argument("--expert_ids", nargs="*", default=None)

    predict = subparsers.add_parser(
        "predict", help="Score one cached novice feature sequence"
    )
    predict.add_argument("--checkpoint", required=True)
    predict.add_argument("--features", required=True)
    predict.add_argument("--fps", type=float, default=25.0)
    predict.add_argument("--expert_features_dir", default=None)
    predict.add_argument("--expert_annotations_dir", default=None)
    predict.add_argument("--expert_ids", nargs="*", default=None)
    predict.add_argument("--out", required=True)
    predict.add_argument("--device", default="cuda")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "dry-run":
        config = load_config(args.config, args.variant)
        _override_expert_ids(config, args.expert_ids)
        dry_run(config, args.seed)
        return
    if args.command == "run":
        config = load_config(args.config, args.variant)
        _override_expert_ids(config, args.expert_ids)
        run_experiment(config, args.variant, args.fold, args.seed, args.device)
        return
    if args.command == "predict":
        report = predict_video(
            checkpoint_path=args.checkpoint,
            query_features_path=args.features,
            output_path=args.out,
            fps=args.fps,
            requested_device=args.device,
            expert_features_dir=args.expert_features_dir,
            expert_annotations_dir=args.expert_annotations_dir,
            expert_ids=args.expert_ids,
        )
        print(
            f"{report['video_id']}: {report['n_detected_events']} events, "
            f"score={report['procedure_score']} -> {args.out}"
        )
        return

    base = load_config(args.config)
    variants = args.variants or base["available_variants"]
    seeds = args.seeds or [
        int(value) for value in base.get("matrix", {}).get("seeds", [0, 1, 2])
    ]
    folds = args.folds or list(range(int(base.get("split", {}).get("folds", 5))))
    commands = _matrix_commands(
        os.path.abspath(args.config), variants, folds, seeds, args.device
    )
    if args.print_only:
        print("\n".join(commands))
        return
    for variant in variants:
        config = load_config(args.config, variant)
        _override_expert_ids(config, args.expert_ids)
        for seed in seeds:
            for fold in folds:
                run_experiment(config, variant, fold, seed, args.device)


if __name__ == "__main__":
    main()
