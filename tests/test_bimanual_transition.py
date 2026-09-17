import json
from pathlib import Path

import numpy as np
import pytest
import torch

from bimanual_transition_data import (
    ANOMALY_TYPES,
    HANDS,
    TRANSITION_TYPES,
    FactorVocabulary,
    audit_annotation_vocabulary,
    build_transitions,
    load_group_map,
    load_video_example,
    make_grouped_folds,
)
from bimanual_transition_metrics import (
    aggregate_transition_predictions,
    average_precision,
    evaluate_event_predictions,
    hierarchical_probabilities,
    hierarchical_decision,
    select_f1_threshold,
)
from bimanual_transition_model import (
    BimanualTransitionModel,
    LongContextObservationModel,
    decode_phase_states,
    make_transition_batch,
    observation_loss,
)
from step2_bimanual_transition import load_config, predict_video, run_experiment
from summarize_bimanual_ablation import aggregate as summarize_ablation
from summarize_bimanual_ablation import discover as discover_ablation_runs
from procedure_scoring import evaluate_procedure_scores, score_procedure
from evaluate_fully_predicted import (
    _select_threshold,
    _score_video,
    _state_decision_video,
    _validate_declared_video_ids,
    _validate_prediction_provenance,
)
from evaluate_fully_predicted import _latent_state_count
from apply_hact_two_state_filter import _filter_rows as _filter_two_state_rows


def _write_video(tmp_path, video_id="p1_take1"):
    features_dir = tmp_path / "features"
    annotations_dir = tmp_path / "annotations"
    features_dir.mkdir()
    annotations_dir.mkdir()
    frame_ids = np.arange(24, dtype=np.int64)
    features = np.random.default_rng(0).normal(size=(24, 8)).astype(np.float32)
    np.savez(
        features_dir / f"{video_id}_features.npz",
        features=features,
        frame_ids=frame_ids,
    )
    annotation = {
        "video_id": video_id,
        "fps": 12.0,
        "events": [
            {
                "event_id": "L1",
                "hand": "Left_hand",
                "start_frame": 2,
                "contact_onset_frame": 4,
                "end_frame": 8,
                "verb": "hold",
                "target_object_name": "housing",
                "tool_object_name": "",
                "anomaly_label": "normal",
                "has_anomaly": False,
            },
            {
                "event_id": "R1",
                "hand": "Right_hand",
                "start_frame": 3,
                "contact_onset_frame": 5,
                "end_frame": 9,
                "verb": "loosen",
                "target_object_name": "screw",
                "tool_object_name": "wrong_driver",
                "anomaly_label": "error_wrong_tool",
                "has_anomaly": True,
            },
            {
                "event_id": "R2",
                "hand": "Right_hand",
                "start_frame": 12,
                "contact_onset_frame": 14,
                "end_frame": 18,
                "verb": "loosen",
                "target_object_name": "screw",
                "tool_object_name": "driver",
                "anomaly_label": "recovery",
                "has_anomaly": False,
            },
        ],
    }
    annotation_path = annotations_dir / f"{video_id}_parsed_annotations.json"
    annotation_path.write_text(json.dumps(annotation), encoding="utf-8")
    return features_dir, annotations_dir, annotation_path


def test_anomalous_events_keep_observed_semantic_and_phase_targets(tmp_path):
    features_dir, annotations_dir, annotation_path = _write_video(tmp_path)
    vocab = FactorVocabulary.build([str(annotation_path)])
    example = load_video_example(
        "p1_take1", str(features_dir), str(annotations_dir), vocab
    )

    wrong_tool = next(event for event in example.events if event.event_id == "R1")
    assert wrong_tool.anomaly == ANOMALY_TYPES.index("error_wrong_tool")
    assert np.all(example.targets.verb[1, 3:10] == wrong_tool.verb)
    assert np.all(example.targets.part[1, 3:10] == wrong_tool.part)
    assert example.targets.phase[1, 3] == 1
    assert example.targets.phase[1, 5] == 2

    recovery = next(event for event in example.events if event.event_id == "R2")
    assert recovery.anomaly == 0
    assert recovery.recovery is True


def test_transitions_normalize_each_action_and_simultaneous_time_group(tmp_path):
    features_dir, annotations_dir, annotation_path = _write_video(tmp_path)
    vocab = FactorVocabulary.build([str(annotation_path)])
    example = load_video_example(
        "p1_take1", str(features_dir), str(annotations_dir), vocab
    )
    transitions = build_transitions(
        example.video_id, example.fps, example.frame_ids, example.events, True
    )
    assert len(transitions) == 9
    for event_id in {item.event_id for item in transitions}:
        assert np.isclose(
            sum(item.event_weight for item in transitions if item.event_id == event_id),
            1.0,
        )

    simultaneous_events = [
        event.__class__(
            event_id=f"same_{event.hand}",
            hand=event.hand,
            start_frame=2,
            onset_frame=4,
            end_frame=8,
            verb=event.verb,
            part=event.part,
            tool=event.tool,
            anomaly=0,
            recovery=False,
        )
        for event in example.events[:2]
    ]
    simultaneous = build_transitions(
        example.video_id, example.fps, example.frame_ids, simultaneous_events, True
    )
    starts = [item for item in simultaneous if item.frame == 2]
    assert len(starts) == 2
    assert all(item.has_prev_global is False for item in starts)
    assert all(item.has_prev_cross is False for item in starts)
    assert np.isclose(sum(item.time_group_weight for item in starts), 1.0)

    onsets = [item for item in simultaneous if item.frame == 4]
    assert all(item.has_prev_cross is True for item in onsets)
    assert all(np.isclose(item.delta_cross, 2.0 / example.fps) for item in onsets)


def test_participant_grouped_folds_have_no_identity_leakage():
    videos = ["p1_a", "p1_b", "p2_a", "p3_a", "p4_a", "p5_a"]
    groups = {video: video.split("_")[0] for video in videos}
    folds = make_grouped_folds(videos, groups, n_folds=3, seed=7)
    for fold in folds:
        train = set(fold["train_groups"])
        val = set(fold["val_groups"])
        test = set(fold["test_groups"])
        assert not train & val
        assert not train & test
        assert not val & test


def test_explicit_participant_map_must_cover_every_video(tmp_path):
    path = tmp_path / "groups.json"
    path.write_text(json.dumps({"video_to_group": {"p1_a": "p1"}}), encoding="utf-8")
    with pytest.raises(ValueError, match="missing 1 video"):
        load_group_map(["p1_a", "p2_a"], str(path))


def test_fixed_vocabulary_never_adapts_to_worker_labels(tmp_path):
    verbs = tmp_path / "verbs.txt"
    nouns = tmp_path / "nouns.txt"
    verbs.write_text("hold\n", encoding="utf-8")
    nouns.write_text("housing\n", encoding="utf-8")
    annotation = tmp_path / "worker.json"
    annotation.write_text(
        json.dumps(
            {
                "events": [
                    {
                        "verb": "held_out_verb",
                        "target_object_name": "housing",
                        "tool_object_name": "",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    vocab = FactorVocabulary.build_from_ontology(str(verbs), str(nouns))
    assert "held_out_verb" not in vocab.verbs
    assert vocab.decode("verb", vocab.encode("verb", "held_out_verb")) == "<unk>"
    unknown = audit_annotation_vocabulary([str(annotation)], vocab)
    assert unknown["verb"] == {"held_out_verb": 1}


def _score_requirements():
    def step(group, index, verb, part):
        return {
            "group_index": group,
            "step_index": index,
            "step_id": index,
            "alternatives": [{"verb": verb, "part": part, "label": f"{verb}:{part}"}],
            "tool": 0,
            "min_repeats": 1,
            "max_repeats": 1,
        }

    return {
        "procedure_name": "test",
        "groups": [
            {"group_index": 0, "name": "A", "steps": [step(0, 0, 1, 1), step(0, 1, 2, 2)]},
            {"group_index": 1, "name": "B", "steps": [step(1, 0, 3, 3)]},
        ],
    }


def test_procedure_score_respects_parallel_groups_and_penalizes_extra_events():
    events = [
        {"event_id": "b", "start_frame": 1, "verb": 3, "part": 3, "anomaly_score": 0.0},
        {"event_id": "a1", "start_frame": 2, "verb": 1, "part": 1, "anomaly_score": 0.0},
        {"event_id": "a2", "start_frame": 3, "verb": 2, "part": 2, "anomaly_score": 0.0},
    ]
    complete = score_procedure(events, _score_requirements())
    assert complete["procedure_score"] == 100.0
    assert complete["completion"] == 1.0

    with_extra = score_procedure(
        events
        + [
            {
                "event_id": "extra",
                "start_frame": 4,
                "verb": 9,
                "part": 9,
                "anomaly_score": 0.0,
            }
        ],
        _score_requirements(),
    )
    assert with_extra["procedure_score"] == 75.0
    assert with_extra["extra_events_count"] == 1


def test_procedure_score_combines_completion_and_execution_without_weights():
    events = [
        {"event_id": "a1", "start_frame": 1, "verb": 1, "part": 1, "anomaly_score": 0.0},
        {"event_id": "a2", "start_frame": 2, "verb": 2, "part": 2, "anomaly_score": 1.0},
        {"event_id": "b", "start_frame": 3, "verb": 3, "part": 3, "anomaly_score": 0.0},
    ]
    result = score_procedure(events, _score_requirements())
    assert np.isclose(result["procedure_score"], 200.0 / 3.0)
    assert result["completion"] == 1.0
    assert np.isclose(result["execution_quality"], 2.0 / 3.0)

    metrics = evaluate_procedure_scores(
        [
            {"predicted_score": 80.0, "reference_score": 100.0},
            {"predicted_score": 50.0, "reference_score": 60.0},
            {"predicted_score": 20.0, "reference_score": 30.0},
        ]
    )
    assert np.isclose(metrics["mae"], 40.0 / 3.0)
    assert metrics["spearman"] == 1.0


def test_procedure_score_absorbs_sop_authorized_corrective_repeat():
    requirements = _score_requirements()
    requirements["groups"][0]["steps"][0]["max_repeats"] = 2
    events = [
        {"event_id": "a1", "start_frame": 1, "verb": 1, "part": 1, "anomaly_score": 0.0},
        {"event_id": "repair", "start_frame": 2, "verb": 1, "part": 1, "anomaly_score": 0.0},
        {"event_id": "a2", "start_frame": 3, "verb": 2, "part": 2, "anomaly_score": 0.0},
        {"event_id": "b", "start_frame": 4, "verb": 3, "part": 3, "anomaly_score": 0.0},
    ]
    result = score_procedure(events, requirements)
    assert result["procedure_score"] == 100.0
    assert result["matched_optional_repeats"] == 1
    assert result["extra_events_count"] == 0


def test_hierarchical_probabilities_mask_unsupported_subtypes():
    probabilities = hierarchical_probabilities(
        np.array([0.0, 0.0]),
        np.zeros(len(ANOMALY_TYPES) - 1),
        supported_types=[True, False, False, False, True, False],
    )
    assert np.isclose(probabilities.sum(), 1.0)
    assert np.isclose(probabilities[0], 0.5)
    assert np.all(probabilities[[2, 3, 4, 6]] == 0.0)
    assert np.isclose(probabilities[1] + probabilities[5], 0.5)


def test_recovery_state_probability_is_non_error_mass():
    probabilities = hierarchical_probabilities(
        np.log(np.array([0.2, 0.3, 0.5])),
        np.zeros(len(ANOMALY_TYPES) - 1),
    )
    assert np.isclose(probabilities.sum(), 1.0)
    assert np.isclose(probabilities[0], 0.7)
    assert np.isclose(probabilities[1:].sum(), 0.3)


def test_binary_threshold_precedes_conditional_anomaly_type():
    probabilities = np.array([0.4, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
    assert hierarchical_decision(probabilities, threshold=0.5) == 1
    assert hierarchical_decision(probabilities, threshold=0.7) == 0
    rows = [
        {
            "target": 1,
            "anomaly_score": 0.6,
            "probabilities": probabilities.tolist(),
        }
    ]
    metrics = evaluate_event_predictions(rows, ANOMALY_TYPES, threshold=0.5)
    assert metrics["per_type"]["error_temporal"]["f1"] == 1.0


def test_phase_decoder_enforces_the_annotated_cycle():
    logits = np.array(
        [
            [4.0, 0.0, 0.0],
            [
                0.0,
                4.0,
                8.0,
            ],  # direct idle->interaction has higher local score but is illegal
            [0.0, 1.0, 5.0],
            [5.0, 0.0, 1.0],
        ]
    )
    path = decode_phase_states(logits)
    assert path.tolist() == [0, 1, 2, 0]


def test_synchronous_bimanual_tokens_are_order_free_within_a_time_group():
    tokens = torch.tensor([[1.0, 3.0], [3.0, 1.0], [8.0, 4.0]])
    frames = torch.tensor([10, 10, 12])
    groups, inverse = BimanualTransitionModel._group_tokens(tokens, frames)
    observed = groups.index_select(0, inverse)
    assert torch.equal(groups, torch.tensor([[2.0, 2.0], [8.0, 4.0]]))
    assert torch.equal(observed[0], observed[1])


def test_history_context_is_strictly_causal(tmp_path):
    _, _, annotation_path = _write_video(tmp_path)
    vocab = FactorVocabulary.build([str(annotation_path)])
    model = BimanualTransitionModel(
        visual_dim=16,
        vocab=vocab,
        model_dim=8,
        num_layers=1,
        num_heads=2,
        dropout=0.0,
    )
    model.eval()
    tokens = torch.randn(4, 8)
    frames = torch.tensor([1, 2, 3, 4])
    original, _ = model._encode_groups(tokens, frames)

    # The context paired with frame 3 must depend only on frames 1--2.
    # Perturbing frame 3 itself and every later token must not change it.
    changed_tokens = tokens.clone()
    changed_tokens[2:] += 100.0
    changed, _ = model._encode_groups(changed_tokens, frames)
    assert torch.allclose(original[:3], changed[:3], atol=1e-6)


def test_async_cross_hand_attention_is_strict_and_preserves_fallback(tmp_path):
    features_dir, annotations_dir, annotation_path = _write_video(tmp_path)
    vocab = FactorVocabulary.build([str(annotation_path)])
    example = load_video_example(
        "p1_take1", str(features_dir), str(annotations_dir), vocab
    )
    records = build_transitions(
        example.video_id, example.fps, example.frame_ids, example.events, True
    )
    hidden = torch.randn(len(example.frame_ids), 4)
    batch = make_transition_batch(records, hidden, torch.device("cpu"))
    model = BimanualTransitionModel(
        visual_dim=8,
        vocab=vocab,
        model_dim=8,
        num_layers=1,
        num_heads=2,
        dropout=0.0,
        cross_hand_fusion="gated",
    )
    model.eval()
    tokens = torch.randn(len(batch), 8)
    self_history = torch.randn_like(tokens)
    original, gate, available, _ = model._cross_hand_context(
        tokens, self_history, batch
    )

    # R1 onset at frame 5 may use only left-hand events at frames 2 and 4.
    target = next(
        index
        for index, record in enumerate(records)
        if record.event_id == "R1" and record.frame == 5
    )
    valid = (batch.hand != batch.hand[target]) & (batch.frames < batch.frames[target])
    changed_tokens = tokens.clone()
    changed_tokens[~valid] += 100.0
    changed, _, _, _ = model._cross_hand_context(
        changed_tokens, self_history, batch
    )
    assert available[target]
    assert torch.allclose(original[target], changed[target], atol=1e-6)

    # Before the opposite hand has produced an event, fusion is an exact
    # per-hand fallback (including a zero gate).
    first = int(torch.argmin(batch.frames).item())
    assert not available[first]
    assert gate[first] == 0
    assert torch.equal(original[first], self_history[first])


def test_role_fusion_preserves_left_right_assignment(tmp_path):
    _, _, annotation_path = _write_video(tmp_path)
    vocab = FactorVocabulary.build([str(annotation_path)])
    width = 8
    model = BimanualTransitionModel(
        visual_dim=16,
        vocab=vocab,
        model_dim=width,
        num_layers=1,
        num_heads=2,
        dropout=0.0,
        use_role_fusion=True,
    )

    class SelectLeft(torch.nn.Module):
        def forward(self, pair):
            return pair[..., :width]

    model.role_fusion = SelectLeft()
    left = torch.arange(width, dtype=torch.float32)
    right = torch.arange(width, dtype=torch.float32) + 20
    frames = torch.tensor([10, 10])
    hands = torch.tensor([0, 1])
    assigned, _ = model._role_group_tokens(
        torch.stack((left, right)), frames, hands
    )
    swapped, _ = model._role_group_tokens(
        torch.stack((right, left)), frames, hands
    )
    assert torch.equal(assigned[0], left)
    assert torch.equal(swapped[0], right)
    assert not torch.equal(assigned, swapped)


def test_role_factorized_model_uses_joint_marks_and_per_hand_clock(tmp_path):
    features_dir, annotations_dir, annotation_path = _write_video(tmp_path)
    vocab = FactorVocabulary.build([str(annotation_path)])
    example = load_video_example(
        "p1_take1", str(features_dir), str(annotations_dir), vocab
    )
    records = build_transitions(
        example.video_id, example.fps, example.frame_ids, example.events, True
    )
    hidden = torch.randn(len(example.frame_ids), 4)
    batch = make_transition_batch(records, hidden, torch.device("cpu"))
    model = BimanualTransitionModel(
        visual_dim=8,
        vocab=vocab,
        model_dim=8,
        num_layers=1,
        num_heads=2,
        dropout=0.0,
        use_role_fusion=True,
        timing_clock="hand",
    )
    model.eval()

    delta, support = model._timing_values(batch)
    assert model.uses_joint_timeline
    assert torch.equal(delta, batch.delta_hand)
    assert torch.equal(support, batch.has_prev_hand)

    # The first right-hand event follows a left-hand event globally, but it is
    # unsupported by the right-hand renewal clock and must receive no time NLL.
    first_right = next(
        index
        for index, record in enumerate(records)
        if record.event_id == "R1"
        and record.transition == TRANSITION_TYPES.index("start")
    )
    assert batch.has_prev_global[first_right]
    assert not batch.has_prev_hand[first_right]
    outputs = model(batch)
    assert outputs["time_nll"][first_right] == 0
    assert outputs["survival_surprise"][first_right] == 0
    assert model.checkpoint_config()["timing_clock"] == "hand"


def test_competing_clock_ties_hand_mark_to_cause_specific_intensity(tmp_path):
    features_dir, annotations_dir, annotation_path = _write_video(tmp_path)
    vocab = FactorVocabulary.build([str(annotation_path)])
    example = load_video_example(
        "p1_take1", str(features_dir), str(annotations_dir), vocab
    )
    records = build_transitions(
        example.video_id, example.fps, example.frame_ids, example.events, True
    )
    hidden = torch.randn(len(example.frame_ids), 4)
    batch = make_transition_batch(records, hidden, torch.device("cpu"))
    model = BimanualTransitionModel(
        visual_dim=8,
        vocab=vocab,
        model_dim=8,
        num_layers=1,
        num_heads=2,
        dropout=0.0,
        use_role_fusion=True,
        timing_clock="competing",
    )
    model.eval()

    assert model.time_head.out_features == len(model.missing_hand_tokens)
    first_right = next(
        index
        for index, record in enumerate(records)
        if record.event_id == "R1"
        and record.transition == TRANSITION_TYPES.index("start")
    )
    assert batch.has_prev_global[first_right]
    outputs = model(batch)
    assert outputs["survival_surprise"][first_right] > 0
    expected_hand_logits = (
        torch.nn.functional.softplus(outputs["time_parameters"]) + 1e-4
    ).log()
    assert torch.allclose(outputs["mark_logits"]["hand"], expected_hand_logits)
    assert model.checkpoint_config()["timing_clock"] == "competing"


def test_explicit_recovery_state_has_three_competing_states(tmp_path):
    features_dir, annotations_dir, annotation_path = _write_video(tmp_path)
    vocab = FactorVocabulary.build([str(annotation_path)])
    example = load_video_example(
        "p1_take1", str(features_dir), str(annotations_dir), vocab
    )
    records = build_transitions(
        example.video_id, example.fps, example.frame_ids, example.events, True
    )
    batch = make_transition_batch(
        records, torch.randn(len(example.frame_ids), 4), torch.device("cpu")
    )
    model = BimanualTransitionModel(
        visual_dim=8,
        vocab=vocab,
        model_dim=8,
        num_layers=1,
        num_heads=2,
        dropout=0.0,
        use_recovery_state=True,
    )
    outputs = model(batch)
    assert outputs["binary_logits"].shape[-1] == 3
    assert model.checkpoint_config()["use_recovery_state"] is True
    counts = torch.ones(len(ANOMALY_TYPES) + 1)
    loss = model.anomaly_loss(outputs, batch, counts)
    loss.backward()
    assert torch.isfinite(loss)


def test_recovery_log_odds_gate_is_identity_at_initialization(tmp_path):
    features_dir, annotations_dir, annotation_path = _write_video(tmp_path)
    vocab = FactorVocabulary.build([str(annotation_path)])
    example = load_video_example(
        "p1_take1", str(features_dir), str(annotations_dir), vocab
    )
    records = build_transitions(
        example.video_id, example.fps, example.frame_ids, example.events, True
    )
    batch = make_transition_batch(
        records, torch.randn(len(example.frame_ids), 4), torch.device("cpu")
    )
    model = BimanualTransitionModel(
        visual_dim=8,
        vocab=vocab,
        model_dim=8,
        num_layers=1,
        num_heads=2,
        dropout=0.0,
        use_recovery_gate=True,
    )
    model.eval()
    model.recovery_gate_enabled = False
    base = model(batch)["binary_logits"]
    model.recovery_gate_enabled = True
    identity = model(batch)["binary_logits"]
    assert torch.equal(base, identity)

    counts = torch.ones(len(ANOMALY_TYPES) + 1)
    outputs = model(batch)
    loss = model.anomaly_loss(outputs, batch, counts, recovery_only=True)
    loss.backward()
    assert torch.isfinite(loss)
    assert model.checkpoint_config()["use_recovery_gate"] is True


def test_exponential_clock_keeps_an_independent_hand_mark(tmp_path):
    features_dir, annotations_dir, annotation_path = _write_video(tmp_path)
    vocab = FactorVocabulary.build([str(annotation_path)])
    example = load_video_example(
        "p1_take1", str(features_dir), str(annotations_dir), vocab
    )
    records = build_transitions(
        example.video_id, example.fps, example.frame_ids, example.events, True
    )
    batch = make_transition_batch(
        records, torch.randn(len(example.frame_ids), 4), torch.device("cpu")
    )
    model = BimanualTransitionModel(
        visual_dim=8,
        vocab=vocab,
        model_dim=8,
        num_layers=1,
        num_heads=2,
        dropout=0.0,
        use_role_fusion=True,
        timing_clock="exponential",
    )
    model.eval()

    outputs = model(batch)
    assert model.time_head.out_features == 1
    assert outputs["mark_logits"]["hand"].shape[-1] == len(HANDS)
    rate = torch.nn.functional.softplus(outputs["time_parameters"].squeeze(-1)) + 1e-4
    expected_survival = rate * batch.delta_global * batch.has_prev_global
    assert torch.allclose(outputs["survival_surprise"], expected_survival)
    assert model.checkpoint_config()["timing_clock"] == "exponential"


def test_correction_state_is_causal_and_trainable(tmp_path):
    features_dir, annotations_dir, annotation_path = _write_video(tmp_path)
    vocab = FactorVocabulary.build([str(annotation_path)])
    example = load_video_example(
        "p1_take1", str(features_dir), str(annotations_dir), vocab
    )
    records = build_transitions(
        example.video_id, example.fps, example.frame_ids, example.events, True
    )
    batch = make_transition_batch(
        records, torch.randn(len(example.frame_ids), 4), torch.device("cpu")
    )
    model = BimanualTransitionModel(
        visual_dim=8,
        vocab=vocab,
        model_dim=8,
        num_layers=1,
        num_heads=2,
        dropout=0.0,
        use_role_fusion=True,
        timing_clock="competing",
        use_correction_state=True,
    )

    outputs = model(batch)
    assert outputs["correction_gate"].shape == (len(batch),)
    assert torch.all((outputs["correction_gate"] > 0) & (outputs["correction_gate"] < 1))
    outputs["binary_logits"].sum().backward()
    update = model.anomaly_head["state_update"]
    assert update.weight_hh.grad is not None
    assert model.checkpoint_config()["use_correction_state"] is True


def test_recurrent_execution_state_uses_fixed_bimanual_belief_slots(tmp_path):
    features_dir, annotations_dir, annotation_path = _write_video(tmp_path)
    vocab = FactorVocabulary.build([str(annotation_path)])
    example = load_video_example(
        "p1_take1", str(features_dir), str(annotations_dir), vocab
    )
    records = build_transitions(
        example.video_id, example.fps, example.frame_ids, example.events, True
    )
    batch = make_transition_batch(
        records, torch.randn(len(example.frame_ids), 4), torch.device("cpu")
    )
    model = BimanualTransitionModel(
        visual_dim=8,
        vocab=vocab,
        model_dim=8,
        num_layers=1,
        num_heads=2,
        dropout=0.0,
        use_role_fusion=True,
        use_recovery_state=True,
        factorized_recovery_state=True,
        use_correction_state=True,
        correction_use_state_belief=True,
    )
    outputs = model(batch)
    assert outputs["binary_logits"].shape == (len(batch), 3)
    assert outputs["base_error_logits"].shape == (len(batch), 2)
    assert outputs["conditional_recovery_logits"].shape == (len(batch), 2)
    assert torch.allclose(
        outputs["binary_logits"].logsumexp(dim=-1),
        torch.zeros(len(batch)),
        atol=1e-6,
    )
    assert model.anomaly_head["state_adapter"][0].in_features == 2 * 8 + 2 * 3
    outputs["binary_logits"].sum().backward()
    assert model.anomaly_head["state_adapter"][0].weight.grad is not None
    assert model.anomaly_head["state_error_update"].weight_hh.grad is not None
    assert model.checkpoint_config()["correction_use_state_belief"] is True
    assert model.checkpoint_config()["factorized_recovery_state"] is True
    model.zero_grad(set_to_none=True)
    model.correction_enabled = False
    base_outputs = model(batch)
    loss = model.anomaly_loss(
        base_outputs, batch, torch.ones(len(ANOMALY_TYPES) + 1)
    )
    loss.backward()
    assert torch.isfinite(loss)
    assert model.anomaly_head["conditional_recovery"].weight.grad is not None


def test_factorized_state_head_does_not_perturb_shared_initialization(tmp_path):
    _, _, annotation_path = _write_video(tmp_path)
    vocab = FactorVocabulary.build([str(annotation_path)])

    def build(factorized):
        torch.manual_seed(17)
        model = BimanualTransitionModel(
            visual_dim=8,
            vocab=vocab,
            model_dim=8,
            num_layers=1,
            num_heads=2,
            dropout=0.0,
            use_recovery_state=True,
            factorized_recovery_state=factorized,
        )
        next_random = torch.rand(4)
        shared = {
            name: value.detach().clone()
            for name, value in model.named_parameters()
            if not name.startswith("anomaly_head.")
        }
        return shared, next_random

    flat, flat_random = build(False)
    factorized, factorized_random = build(True)
    assert flat.keys() == factorized.keys()
    assert all(torch.equal(flat[name], factorized[name]) for name in flat)
    assert torch.equal(flat_random, factorized_random)


def test_rare_subtypes_can_be_excluded_from_conditional_training(tmp_path):
    _, _, annotation_path = _write_video(tmp_path)
    vocab = FactorVocabulary.build([str(annotation_path)])
    model = BimanualTransitionModel(
        visual_dim=16,
        vocab=vocab,
        model_dim=8,
        num_layers=1,
        num_heads=2,
        minimum_subtype_count=2.0,
    )
    counts = torch.zeros(len(ANOMALY_TYPES))
    counts[0] = 100
    counts[1] = 1
    counts[5] = 3
    model.set_anomaly_support(counts)
    support = model.supported_anomaly_types.tolist()
    assert support[0] is False
    assert support[4] is True


def test_observation_and_transition_models_support_full_backward(tmp_path):
    features_dir, annotations_dir, annotation_path = _write_video(tmp_path)
    vocab = FactorVocabulary.build([str(annotation_path)])
    example = load_video_example(
        "p1_take1", str(features_dir), str(annotations_dir), vocab
    )
    observation = LongContextObservationModel(
        input_dim=8, vocab=vocab, hidden_dim=16, max_dilation=4, dropout=0.0
    )
    features = torch.from_numpy(example.features)
    hidden, logits = observation(features)
    targets = {
        "verb": torch.from_numpy(example.targets.verb),
        "part": torch.from_numpy(example.targets.part),
        "tool": torch.from_numpy(example.targets.tool),
        "phase": torch.from_numpy(example.targets.phase),
    }
    counts = {
        name: torch.bincount(
            target.reshape(-1), minlength=logits["left"][name].shape[-1]
        )
        for name, target in targets.items()
    }
    obs_loss = observation_loss(logits, targets, counts)
    obs_loss.backward(retain_graph=True)
    assert torch.isfinite(obs_loss)

    records = build_transitions(
        example.video_id, example.fps, example.frame_ids, example.events, True
    )
    batch = make_transition_batch(records, hidden.detach(), torch.device("cpu"))
    transition = BimanualTransitionModel(
        visual_dim=32,
        vocab=vocab,
        model_dim=16,
        num_layers=1,
        num_heads=4,
        dropout=0.0,
    )
    outputs = transition(batch, expert_batches=[batch], sop_marks=[])
    assert outputs["anomaly_logits"].shape == (len(records), len(ANOMALY_TYPES))
    assert torch.all(outputs["time_nll"][~batch.has_prev_global] == 0)
    assert torch.all(outputs["survival_surprise"][~batch.has_prev_global] == 0)
    loss = transition.normative_loss(outputs, batch) + transition.anomaly_loss(
        outputs, batch, torch.ones(len(ANOMALY_TYPES))
    )
    loss.backward()
    assert torch.isfinite(loss)


def test_survival_surprise_increases_when_transition_is_overdue():
    parameters = torch.tensor([[0.0, 0.0], [0.0, 0.0]])
    _, surprise = BimanualTransitionModel._time_terms(
        parameters, torch.tensor([0.5, 4.0])
    )
    assert surprise[1] > surprise[0]


def test_metrics_aggregate_phase_transitions_once():
    rows = []
    for frame in (1, 2, 3):
        rows.append(
            {
                "video_id": "v",
                "event_id": "e",
                "event_start_frame": 1,
                "event_end_frame": 3,
                "frame": frame,
                "hand": 0,
                "target": ANOMALY_TYPES.index("error_temporal"),
                "recovery": False,
                "logits": [0.0, 4.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                "evidence": {"time_survival_surprise": float(frame)},
            }
        )
    events = aggregate_transition_predictions(rows, ANOMALY_TYPES)
    assert len(events) == 1
    assert [point["frame"] for point in events[0]["transition_predictions"]] == [
        1,
        2,
        3,
    ]
    metrics = evaluate_event_predictions(events, ANOMALY_TYPES, threshold=0.5)
    assert metrics["n_events"] == 1
    assert metrics["anomaly_f1"] == 1.0


def test_aggregation_retains_direct_recovery_state_posteriors():
    rows = [
        {
            "video_id": "v",
            "event_id": "e",
            "event_start_frame": 1,
            "event_end_frame": 3,
            "frame": frame,
            "transition": frame - 1,
            "hand": 0,
            "verb": 0,
            "part": 0,
            "tool": 0,
            "target": 0,
            "recovery": True,
            "logits": [0.0] * len(ANOMALY_TYPES),
            "binary_logits": [0.0, 0.0, 2.0],
            "type_logits": [0.0] * (len(ANOMALY_TYPES) - 1),
            "supported_anomaly_types": [True] * (len(ANOMALY_TYPES) - 1),
            "evidence": {},
        }
        for frame in (1, 2, 3)
    ]
    event = aggregate_transition_predictions(rows, ANOMALY_TYPES)[0]
    assert int(np.argmax(event["state_probabilities"])) == 2
    assert all(
        int(np.argmax(point["state_probabilities"])) == 2
        for point in event["transition_predictions"]
    )


def test_phase_timeline_holds_each_causal_transition_score_until_the_next():
    row = {
        "start_frame": 1,
        "end_frame": 6,
        "anomaly_score": 0.5,
        "transition_predictions": [
            {"frame": 1, "transition": 0, "anomaly_score": 0.1},
            {"frame": 3, "transition": 1, "anomaly_score": 0.8},
            {"frame": 6, "transition": 2, "anomaly_score": 0.2},
        ],
    }
    event_score = _score_video(8, [row], True, "event")
    phase_score = _score_video(8, [row], True, "phase")
    assert np.allclose(event_score[1:7], 0.5)
    assert np.allclose(phase_score[1:3], 0.1)
    assert np.allclose(phase_score[3:6], 0.8)
    assert np.allclose(phase_score[6:7], 0.2)
    assert phase_score[0] == phase_score[7] == 0.0


def test_two_state_filter_is_causal_and_preserves_predicted_support():
    dynamics = {
        "initial": [0.8, 0.2],
        "transition": [[0.9, 0.1], [0.6, 0.4]],
    }

    def rows(future_score):
        return [
            {
                "video_id": "v",
                "event_id": "first",
                "hand": 0,
                "start_frame": 1,
                "end_frame": 5,
                "verb": "insert",
                "anomaly_score": 0.4,
                "transition_predictions": [
                    {"frame": 1, "transition": 0, "anomaly_score": 0.2},
                    {"frame": 3, "transition": 1, "anomaly_score": 0.6},
                ],
            },
            {
                "video_id": "v",
                "event_id": "future",
                "hand": 0,
                "start_frame": 7,
                "end_frame": 9,
                "verb": "tighten",
                "anomaly_score": future_score,
                "transition_predictions": [
                    {"frame": 7, "transition": 0, "anomaly_score": future_score}
                ],
            },
        ]

    low_future = _filter_two_state_rows(rows(0.1), dynamics)
    high_future = _filter_two_state_rows(rows(0.9), dynamics)
    assert low_future[0]["state_probabilities"] == high_future[0][
        "state_probabilities"
    ]
    for source, filtered in zip(rows(0.1), low_future):
        assert (filtered["event_id"], filtered["start_frame"], filtered["end_frame"]) == (
            source["event_id"],
            source["start_frame"],
            source["end_frame"],
        )
        assert filtered["verb"] == source["verb"]
        assert [
            (point["frame"], point["transition"])
            for point in filtered["transition_predictions"]
        ] == [
            (point["frame"], point["transition"])
            for point in source["transition_predictions"]
        ]


def test_state_map_decision_has_no_scalar_alarm_threshold():
    rows = [
        {
            "start_frame": 1,
            "end_frame": 6,
            "state_probabilities": [0.7, 0.2, 0.1],
            "transition_predictions": [
                {"frame": 1, "state_probabilities": [0.6, 0.3, 0.1]},
                {"frame": 3, "state_probabilities": [0.1, 0.8, 0.1]},
                {"frame": 6, "state_probabilities": [0.1, 0.2, 0.7]},
            ],
        }
    ]
    decision = _state_decision_video(8, rows, True, "phase")
    assert not decision[1:3].any()
    assert decision[3:6].all()
    assert not decision[6]


def test_fully_predicted_evaluator_rejects_wrong_declared_fold():
    split = {"val": ["p1_val"], "test": ["p2_test"]}
    selected = {
        "validation_video_ids": ["p1_val"],
        "test_video_ids": ["p3_wrong"],
    }
    with pytest.raises(ValueError, match="wrong test videos"):
        _validate_declared_video_ids("hact", 0, split, selected)


def test_fully_predicted_evaluator_rejects_annotation_fields():
    selected = {
        "validation": [],
        "test": [{"anomaly_score": 0.4, "target": 1}],
    }
    with pytest.raises(ValueError, match="annotation fields"):
        _validate_prediction_provenance("hact", 0, selected)


def test_zero_anomaly_scores_cannot_gain_f1_from_a_zero_threshold():
    frame_threshold = _select_threshold(
        np.asarray([1, 0]), np.asarray([0.0, 0.0])
    )
    assert frame_threshold > 0.0

    threshold = select_f1_threshold([1, 0], [0.0, 0.0])
    assert threshold > 0.0
    rows = [
        {"target": 1, "anomaly_score": 0.0, "probabilities": [1.0, 0, 0, 0, 0, 0, 0]},
        {"target": 0, "anomaly_score": 0.0, "probabilities": [1.0, 0, 0, 0, 0, 0, 0]},
    ]
    assert (
        evaluate_event_predictions(rows, ANOMALY_TYPES, threshold)["anomaly_f1"] == 0.0
    )


def test_average_precision_is_invariant_to_tied_score_order():
    assert average_precision([1, 0], [0.0, 0.0]) == 0.5
    assert average_precision([0, 1], [0.0, 0.0]) == 0.5


def test_cross_validated_metrics_use_each_folds_validation_threshold():
    rows = [
        {
            "target": 0,
            "anomaly_score": 0.6,
            "decision_threshold": 0.7,
            "probabilities": [0.4, 0.6, 0, 0, 0, 0, 0],
        },
        {
            "target": 1,
            "anomaly_score": 0.6,
            "decision_threshold": 0.5,
            "probabilities": [0.4, 0.6, 0, 0, 0, 0, 0],
        },
    ]
    metrics = evaluate_event_predictions(rows, ANOMALY_TYPES, threshold=None)
    assert metrics["threshold_mode"] == "validation_per_fold"
    assert metrics["anomaly_f1"] == 1.0


def test_complete_ablation_run_smoke(tmp_path):
    features_dir = tmp_path / "features_all"
    annotations_dir = tmp_path / "annotations_all"
    output_root = tmp_path / "runs"
    features_dir.mkdir()
    annotations_dir.mkdir()
    rng = np.random.default_rng(4)
    video_ids = [f"p{index}_take" for index in range(1, 7)] + ["expert_1"]
    for index, video_id in enumerate(video_ids):
        frame_ids = np.arange(12, dtype=np.int64)
        features = rng.normal(size=(12, 6)).astype(np.float32)
        np.savez(
            features_dir / f"{video_id}_features.npz",
            features=features,
            frame_ids=frame_ids,
        )
        anomalous = video_id != "expert_1" and index % 2 == 1
        annotation = {
            "video_id": video_id,
            "fps": 6.0,
            "events": [
                {
                    "event_id": "event",
                    "hand": "Left_hand" if index % 2 == 0 else "Right_hand",
                    "start_frame": 1,
                    "contact_onset_frame": 4,
                    "end_frame": 9,
                    "verb": "hold" if index % 2 == 0 else "loosen",
                    "target_object_name": (
                        "gearbox_housing" if index % 2 == 0 else "screw"
                    ),
                    "tool_object_name": (
                        "" if index % 2 == 0 else "torx_screwdriver"
                    ),
                    "anomaly_label": "error_temporal" if anomalous else "normal",
                    "has_anomaly": anomalous,
                }
            ],
        }
        (annotations_dir / f"{video_id}_parsed_annotations.json").write_text(
            json.dumps(annotation), encoding="utf-8"
        )

    root = Path(__file__).resolve().parents[1]
    config = {
        "variant": "full",
        "output_root": str(output_root),
        "data": {
            "features_dir": str(features_dir),
            "annotations_dir": str(annotations_dir),
            "expert_features_dir": str(features_dir),
            "expert_annotations_dir": str(annotations_dir),
            "expert_ids": ["expert_1"],
            "sop_path": str(root / "configs" / "public_impact" / "sop_reassembly_a.json"),
            "verbs_path": str(root / "configs" / "public_impact" / "verbs.txt"),
            "nouns_path": str(root / "configs" / "public_impact" / "nouns.txt"),
            "ontology_path": str(root / "configs" / "public_impact" / "verb_noun_ontology.csv"),
            "objects_path": str(root / "configs" / "public_impact" / "objects.json"),
            "strict_vocabulary": True,
            "groups_json": None,
        },
        "split": {"folds": 3},
        "observation": {"hidden_dim": 8, "max_dilation": 2, "dropout": 0.0},
        "transition": {
            "model_dim": 8,
            "num_layers": 1,
            "num_heads": 2,
            "dropout": 0.0,
            "use_phase_transitions": True,
            "use_history": True,
            "use_bimanual_context": True,
            "use_expert_memory": True,
            "use_sop_memory": False,
            "use_time_likelihood": True,
        },
        "training": {
            "observation_epochs": 1,
            "observation_patience": 1,
            "normative_epochs": 1,
            "normative_patience": 1,
            "anomaly_epochs": 1,
            "anomaly_patience": 1,
        },
        "evaluation": {"event_match_iou": 0.1},
        "augmentation": {"observed_event_view": True},
    }
    metrics = run_experiment(
        config, "full", fold_index=0, seed=0, requested_device="cpu"
    )
    run_dir = output_root / "full" / "seed_0" / "fold_0"
    assert metrics["predicted_event"]["n_events"] >= 1
    assert "foreground_macro_f1_supported" in metrics["observation"]
    assert "f1_at_10" in metrics["observation"]["event_detection"]
    assert "f1_at_25" in metrics["observation"]["event_detection"]
    assert "f1_at_50" in metrics["observation"]["event_detection"]
    assert metrics["procedure_score"]["n_videos"] >= 1
    assert (run_dir / "model.pt").exists()
    assert (run_dir / "metrics.json").exists()
    assert (run_dir / "event_noise_report.json").exists()
    assert (run_dir / "procedure_scores.json").exists()
    report = predict_video(
        checkpoint_path=str(run_dir / "model.pt"),
        query_features_path=str(features_dir / "p1_take_features.npz"),
        output_path=str(tmp_path / "prediction.json"),
        fps=6.0,
        requested_device="cpu",
    )
    assert report["video_id"] == "p1_take"
    assert report["procedure_score"] is not None
    assert report["procedure_assessment"]["available"] is True
    assert (tmp_path / "prediction.json").exists()
    summary = summarize_ablation(
        discover_ablation_runs(str(output_root)), "predicted_event"
    )
    assert summary["variants"]["full"]["fold_runs"] == 1
    assert (
        summary["seed_results"][0]["metrics"]["threshold_mode"] == "validation_per_fold"
    )


def test_role_fusion_interactions_flag_controls_the_fused_terms(tmp_path):
    _, _, annotation_path = _write_video(tmp_path)
    vocab = FactorVocabulary.build([str(annotation_path)])

    def build(interactions):
        return BimanualTransitionModel(
            visual_dim=16,
            vocab=vocab,
            model_dim=8,
            num_layers=1,
            num_heads=2,
            dropout=0.0,
            use_hand_identity=True,
            use_role_fusion=True,
            role_fusion_interactions=interactions,
        )

    with_interactions = build(True)
    without = build(False)
    # Concatenation, difference and product against concatenation alone.
    assert with_interactions.role_fusion[1].in_features == 4 * 8
    assert without.role_fusion[1].in_features == 2 * 8

    # The default is the full model, so existing checkpoints keep their shape.
    assert build(True).role_fusion_interactions is True
    assert BimanualTransitionModel(
        visual_dim=16, vocab=vocab, model_dim=8, num_layers=1, num_heads=2
    ).role_fusion_interactions is True

    # The flag is carried in the metadata a checkpoint is rebuilt from.
    config = dict(without.checkpoint_config())
    config.pop("vocab", None)
    assert config["role_fusion_interactions"] is False
    rebuilt = BimanualTransitionModel(vocab=vocab, **config)
    rebuilt.load_state_dict(without.state_dict())


def test_ungated_execution_state_writes_every_transition(tmp_path):
    _, _, annotation_path = _write_video(tmp_path)
    vocab = FactorVocabulary.build([str(annotation_path)])

    def build(gated):
        return BimanualTransitionModel(
            visual_dim=16,
            vocab=vocab,
            model_dim=8,
            num_layers=1,
            num_heads=2,
            dropout=0.0,
            use_hand_identity=True,
            use_role_fusion=True,
            use_correction_state=True,
            gate_execution_state=gated,
        )

    # The default keeps the evidence gate, so existing checkpoints are unaffected.
    assert build(True).gate_execution_state is True
    assert BimanualTransitionModel(
        visual_dim=16, vocab=vocab, model_dim=8, num_layers=1, num_heads=2
    ).gate_execution_state is True

    ungated = build(False)
    assert ungated.gate_execution_state is False
    # Turning the gate off changes no parameter shape: the two controls are
    # seed-matched and differ only in how the state is written.
    gated = build(True)
    assert {k: v.shape for k, v in gated.state_dict().items()} == {
        k: v.shape for k, v in ungated.state_dict().items()
    }
    config = dict(ungated.checkpoint_config())
    config.pop("vocab", None)
    assert config["gate_execution_state"] is False
    BimanualTransitionModel(vocab=vocab, **config).load_state_dict(ungated.state_dict())
