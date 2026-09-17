"""Models for expert-conditioned bimanual procedural anomaly assessment."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from bimanual_transition_data import (
    ANOMALY_TYPES,
    HANDS,
    PHASE_STATES,
    TRANSITION_TYPES,
    AnnotatedEvent,
    FactorVocabulary,
    TransitionRecord,
)


def _sinusoidal_positions(
    length: int, width: int, device: torch.device
) -> torch.Tensor:
    if length <= 0:
        return torch.empty((0, width), device=device)
    position = torch.arange(length, dtype=torch.float32, device=device).unsqueeze(1)
    scale = torch.exp(
        torch.arange(0, width, 2, dtype=torch.float32, device=device)
        * (-math.log(10000.0) / width)
    )
    encoding = torch.zeros((length, width), dtype=torch.float32, device=device)
    encoding[:, 0::2] = torch.sin(position * scale)
    encoding[:, 1::2] = torch.cos(position * scale[: encoding[:, 1::2].shape[1]])
    return encoding


class DilatedResidualBlock(nn.Module):
    def __init__(self, width: int, dilation: int, dropout: float) -> None:
        super().__init__()
        self.conv = nn.Conv1d(
            width, width, kernel_size=3, padding=dilation, dilation=dilation
        )
        self.norm = nn.LayerNorm(width)
        self.dropout = nn.Dropout(dropout)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        hidden = self.conv(inputs)
        hidden = self.norm(hidden.transpose(1, 2)).transpose(1, 2)
        return inputs + self.dropout(F.gelu(hidden))


class LongContextObservationModel(nn.Module):
    """Long-context TCN with hand-specialized factor and phase heads."""

    FACTORS = ("verb", "part", "tool", "phase")

    def __init__(
        self,
        input_dim: int,
        vocab: FactorVocabulary,
        hidden_dim: int = 256,
        max_dilation: int = 512,
        dropout: float = 0.10,
    ) -> None:
        super().__init__()
        if max_dilation < 1 or max_dilation & (max_dilation - 1):
            raise ValueError("max_dilation must be a positive power of two")
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.max_dilation = int(max_dilation)
        self.dropout_rate = float(dropout)
        self.vocab = vocab

        dilations = []
        dilation = 1
        while dilation <= max_dilation:
            dilations.append(dilation)
            dilation *= 2
        self.projection = nn.Linear(input_dim, hidden_dim)
        self.blocks = nn.ModuleList(
            DilatedResidualBlock(hidden_dim, dilation, dropout)
            for dilation in dilations
        )
        sizes = {
            "verb": vocab.size("verb"),
            "part": vocab.size("part"),
            "tool": vocab.size("tool"),
            "phase": len(PHASE_STATES),
        }
        self.heads = nn.ModuleDict(
            {
                hand: nn.ModuleDict(
                    {
                        factor: nn.Linear(hidden_dim, size)
                        for factor, size in sizes.items()
                    }
                )
                for hand in HANDS
            }
        )

    @property
    def receptive_field(self) -> int:
        return 1 + 2 * sum(2**index for index in range(len(self.blocks)))

    def encode(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim != 2:
            raise ValueError(f"Expected [T,D] features, got {tuple(features.shape)}")
        hidden = self.projection(features).transpose(0, 1).unsqueeze(0)
        for block in self.blocks:
            hidden = block(hidden)
        return hidden.squeeze(0).transpose(0, 1)

    def forward(
        self, features: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, Dict[str, torch.Tensor]]]:
        hidden = self.encode(features)
        logits = {
            hand: {factor: head(hidden) for factor, head in self.heads[hand].items()}
            for hand in HANDS
        }
        return hidden, logits

    def freeze_except_last_blocks(self, count: int) -> None:
        for parameter in self.parameters():
            parameter.requires_grad = False
        if count <= 0:
            return
        for block in self.blocks[-count:]:
            for parameter in block.parameters():
                parameter.requires_grad = True

    def checkpoint_config(self) -> dict:
        return {
            "input_dim": self.input_dim,
            "hidden_dim": self.hidden_dim,
            "max_dilation": self.max_dilation,
            "dropout": self.dropout_rate,
            "vocab": self.vocab.to_dict(),
        }


def balanced_softmax_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    class_counts: torch.Tensor,
    sample_weights: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Balanced Softmax without a manually tuned focal-loss exponent."""

    prior = (
        class_counts.to(device=logits.device, dtype=logits.dtype).clamp_min(1.0).log()
    )
    losses = F.cross_entropy(logits + prior.unsqueeze(0), targets, reduction="none")
    if sample_weights is None:
        return losses.mean()
    weights = sample_weights.to(device=logits.device, dtype=losses.dtype)
    return (losses * weights).sum() / weights.sum().clamp_min(1e-8)


def observation_loss(
    logits: Mapping[str, Mapping[str, torch.Tensor]],
    targets: Mapping[str, torch.Tensor],
    class_counts: Mapping[str, torch.Tensor],
) -> torch.Tensor:
    losses = []
    for hand_index, hand in enumerate(HANDS):
        for factor in LongContextObservationModel.FACTORS:
            losses.append(
                balanced_softmax_loss(
                    logits[hand][factor],
                    targets[factor][hand_index],
                    class_counts[factor],
                )
            )
    return torch.stack(losses).mean()


def decode_phase_states(logits: np.ndarray) -> np.ndarray:
    """Viterbi decode the semantic idle->approach->interaction cycle.

    There are no duration thresholds or tuned transition penalties.  The
    finite-state mask only enforces the meaning of the three annotated phases.
    """

    if logits.ndim != 2 or logits.shape[1] != len(PHASE_STATES):
        raise ValueError(f"Expected [T,{len(PHASE_STATES)}] phase logits")
    if len(logits) == 0:
        return np.empty(0, dtype=np.int64)
    allowed = np.array(
        [
            [True, True, False],  # idle -> idle/approach
            [False, True, True],  # approach -> approach/interaction
            [True, False, True],  # interaction -> idle/interaction
        ],
        dtype=bool,
    )
    scores = logits.astype(np.float64, copy=False)
    dynamic = np.full_like(scores, -np.inf)
    back = np.zeros_like(scores, dtype=np.int64)
    dynamic[0] = scores[0]
    for time_index in range(1, len(scores)):
        for state in range(len(PHASE_STATES)):
            previous = np.flatnonzero(allowed[:, state])
            candidates = dynamic[time_index - 1, previous]
            best = int(previous[int(np.argmax(candidates))])
            dynamic[time_index, state] = (
                dynamic[time_index - 1, best] + scores[time_index, state]
            )
            back[time_index, state] = best
    path = np.zeros(len(scores), dtype=np.int64)
    path[-1] = int(np.argmax(dynamic[-1]))
    for time_index in range(len(scores) - 1, 0, -1):
        path[time_index - 1] = back[time_index, path[time_index]]
    return path


def decode_observed_events(
    frame_ids: np.ndarray,
    logits: Mapping[str, Mapping[str, np.ndarray]],
    vocab: FactorVocabulary,
) -> List[AnnotatedEvent]:
    """Turn phase-state transitions into predicted action events."""

    predicted: List[AnnotatedEvent] = []
    approach_id = PHASE_STATES.index("approach")
    interaction_id = PHASE_STATES.index("interaction")
    idle_id = PHASE_STATES.index("idle")

    for hand_index, hand in enumerate(HANDS):
        phase_path = decode_phase_states(np.asarray(logits[hand]["phase"]))
        start_index: Optional[int] = None
        onset_index: Optional[int] = None
        event_number = 0
        for index in range(len(phase_path) + 1):
            previous = idle_id if index == 0 else int(phase_path[index - 1])
            current = idle_id if index == len(phase_path) else int(phase_path[index])
            if previous == idle_id and current == approach_id:
                start_index = index
                onset_index = None
            elif (
                previous == approach_id
                and current == interaction_id
                and start_index is not None
            ):
                onset_index = index
            elif (
                previous == interaction_id
                and current == idle_id
                and start_index is not None
            ):
                end_index = max(start_index, index - 1)
                if onset_index is None:
                    start_index = None
                    continue
                lo, hi = start_index, end_index + 1
                factor_ids = {}
                for factor in ("verb", "part", "tool"):
                    factor_logits = np.asarray(logits[hand][factor])[lo:hi]
                    factor_ids[factor] = int(np.argmax(factor_logits.mean(axis=0)))
                predicted.append(
                    AnnotatedEvent(
                        event_id=f"pred_{hand}_{event_number}",
                        hand=hand_index,
                        start_frame=int(frame_ids[start_index]),
                        onset_frame=int(frame_ids[onset_index]),
                        end_frame=int(frame_ids[end_index]),
                        verb=factor_ids["verb"],
                        part=factor_ids["part"],
                        tool=factor_ids["tool"],
                        anomaly=0,
                        recovery=False,
                    )
                )
                event_number += 1
                start_index = None
                onset_index = None
    predicted.sort(key=lambda event: (event.start_frame, event.hand, event.event_id))
    return predicted


@dataclass
class TransitionBatch:
    records: Sequence[TransitionRecord]
    visual: torch.Tensor
    frames: torch.Tensor
    hand: torch.Tensor
    transition: torch.Tensor
    verb: torch.Tensor
    part: torch.Tensor
    tool: torch.Tensor
    anomaly: torch.Tensor
    recovery: torch.Tensor
    event_weight: torch.Tensor
    time_weight: torch.Tensor
    delta_global: torch.Tensor
    delta_hand: torch.Tensor
    delta_cross: torch.Tensor
    has_prev_global: torch.Tensor
    has_prev_hand: torch.Tensor
    has_prev_cross: torch.Tensor

    def __len__(self) -> int:
        return len(self.records)


def make_transition_batch(
    records: Sequence[TransitionRecord],
    hidden: torch.Tensor,
    device: torch.device,
) -> TransitionBatch:
    if not records:
        raise ValueError("A transition sequence cannot be empty")
    indices = torch.tensor(
        [item.feature_index for item in records], dtype=torch.long, device=hidden.device
    )
    points = hidden.index_select(0, indices)
    phase_pools = []
    for item in records:
        lo = min(item.pool_start_index, item.pool_end_index)
        hi = max(item.pool_start_index, item.pool_end_index) + 1
        phase_pools.append(hidden[lo:hi].mean(dim=0))
    visual = torch.cat((points, torch.stack(phase_pools)), dim=-1).to(device)

    def tensor(name: str, dtype: torch.dtype) -> torch.Tensor:
        return torch.tensor(
            [getattr(item, name) for item in records], dtype=dtype, device=device
        )

    return TransitionBatch(
        records=records,
        visual=visual,
        frames=tensor("frame", torch.long),
        hand=tensor("hand", torch.long),
        transition=tensor("transition", torch.long),
        verb=tensor("verb", torch.long),
        part=tensor("part", torch.long),
        tool=tensor("tool", torch.long),
        anomaly=tensor("anomaly", torch.long),
        recovery=tensor("recovery", torch.long),
        event_weight=tensor("event_weight", torch.float32),
        time_weight=tensor("time_group_weight", torch.float32),
        delta_global=tensor("delta_global", torch.float32),
        delta_hand=tensor("delta_hand", torch.float32),
        delta_cross=tensor("delta_cross", torch.float32),
        has_prev_global=tensor("has_prev_global", torch.bool),
        has_prev_hand=tensor("has_prev_hand", torch.bool),
        has_prev_cross=tensor("has_prev_cross", torch.bool),
    )


class BimanualTransitionModel(nn.Module):
    """One conditional likelihood for semantics, timing and anomaly evidence."""

    EVIDENCE_NAMES = (
        "hand_surprise",
        "transition_surprise",
        "verb_surprise",
        "part_surprise",
        "tool_surprise",
        "time_nll",
        "time_survival_surprise",
        "visual_residual",
    )
    COORDINATION_EVIDENCE_NAMES = (
        "cross_transition_surprise",
        "cross_verb_surprise",
        "cross_part_surprise",
        "cross_tool_surprise",
        "cross_time_nll",
    )

    def __init__(
        self,
        visual_dim: int,
        vocab: FactorVocabulary,
        model_dim: int = 256,
        num_layers: int = 2,
        num_heads: int = 4,
        dropout: float = 0.10,
        use_history: bool = True,
        use_bimanual_context: bool = True,
        use_expert_memory: bool = True,
        use_sop_memory: bool = True,
        use_time_likelihood: bool = True,
        timing_clock: str = "context",
        use_hand_identity: bool = True,
        use_event_semantics: bool = True,
        use_role_fusion: bool = False,
        role_fusion_interactions: bool = True,
        gate_execution_state: bool = True,
        use_recovery_state: bool = False,
        factorized_recovery_state: bool = False,
        use_recovery_gate: bool = False,
        use_correction_state: bool = False,
        correction_state_mode: str = "joint",
        correction_use_error_trace: bool = False,
        correction_use_state_belief: bool = False,
        anomaly_hidden_dim: Optional[int] = None,
        subtype_loss_weight: float = 1.0,
        minimum_subtype_count: float = 1.0,
        recovery_weight: float = 1.0,
        hard_negative_weight: float = 1.0,
        memory_dropout: float = 0.0,
        cross_hand_fusion: str = "joint",
        cross_hand_dropout: float = 0.0,
        cross_gate_bias: float = -2.0,
        cross_auxiliary_weight: float = 0.25,
        train_visual_during_anomaly: bool = False,
        use_effect_evidence: bool = False,
    ) -> None:
        super().__init__()
        if model_dim % num_heads:
            raise ValueError("model_dim must be divisible by num_heads")
        self.visual_dim = int(visual_dim)
        self.model_dim = int(model_dim)
        self.num_layers = int(num_layers)
        self.num_heads = int(num_heads)
        self.dropout_rate = float(dropout)
        self.use_history = bool(use_history)
        self.use_bimanual_context = bool(use_bimanual_context)
        self.use_expert_memory = bool(use_expert_memory)
        self.use_sop_memory = bool(use_sop_memory)
        self.use_time_likelihood = bool(use_time_likelihood)
        self.timing_clock = str(timing_clock).strip().lower()
        self.use_hand_identity = bool(use_hand_identity)
        self.use_event_semantics = bool(use_event_semantics)
        self.use_role_fusion = bool(use_role_fusion)
        self.role_fusion_interactions = bool(role_fusion_interactions)
        self.gate_execution_state = bool(gate_execution_state)
        self.use_recovery_state = bool(use_recovery_state)
        self.factorized_recovery_state = bool(factorized_recovery_state)
        self.use_recovery_gate = bool(use_recovery_gate)
        self.recovery_gate_enabled = self.use_recovery_gate
        self.use_correction_state = bool(use_correction_state)
        self.correction_enabled = self.use_correction_state
        self.correction_state_mode = str(correction_state_mode).strip().lower()
        self.correction_use_error_trace = bool(correction_use_error_trace)
        self.correction_use_state_belief = bool(correction_use_state_belief)
        self.anomaly_hidden_dim = int(anomaly_hidden_dim or model_dim)
        self.subtype_loss_weight = float(subtype_loss_weight)
        self.minimum_subtype_count = float(minimum_subtype_count)
        self.recovery_weight = float(recovery_weight)
        self.hard_negative_weight = float(hard_negative_weight)
        self.memory_dropout = float(memory_dropout)
        self.cross_hand_fusion = str(cross_hand_fusion).strip().lower()
        self.cross_hand_dropout = float(cross_hand_dropout)
        self.cross_gate_bias = float(cross_gate_bias)
        self.cross_auxiliary_weight = float(cross_auxiliary_weight)
        self.train_visual_during_anomaly = bool(train_visual_during_anomaly)
        self.use_effect_evidence = bool(use_effect_evidence)
        if not 0.0 <= self.memory_dropout < 1.0:
            raise ValueError("memory_dropout must be in [0, 1)")
        if self.timing_clock not in {"context", "hand", "competing", "exponential"}:
            raise ValueError(
                "timing_clock must be one of: context, hand, competing, exponential"
            )
        if self.cross_hand_fusion not in {
            "joint",
            "residual",
            "gated",
            "evidence",
        }:
            raise ValueError(
                "cross_hand_fusion must be one of: joint, residual, gated, evidence"
            )
        if self.correction_state_mode not in {"joint", "independent"}:
            raise ValueError(
                "correction_state_mode must be one of: joint, independent"
            )
        if self.use_recovery_state and self.use_recovery_gate:
            raise ValueError(
                "use_recovery_state and use_recovery_gate are alternative "
                "recovery formulations"
            )
        if self.factorized_recovery_state and not self.use_recovery_state:
            raise ValueError(
                "factorized_recovery_state requires use_recovery_state=True"
            )
        if self.correction_use_error_trace and self.correction_use_state_belief:
            raise ValueError(
                "The error trace and the full execution-state belief are "
                "alternative recurrent state inputs"
            )
        if self.correction_use_state_belief and not (
            self.use_correction_state and self.use_recovery_state
        ):
            raise ValueError(
                "correction_use_state_belief requires correction and recovery states"
            )
        if not 0.0 <= self.cross_hand_dropout < 1.0:
            raise ValueError("cross_hand_dropout must be in [0, 1)")
        if self.cross_auxiliary_weight < 0.0:
            raise ValueError("cross_auxiliary_weight must be non-negative")
        if self.subtype_loss_weight < 0.0:
            raise ValueError("subtype_loss_weight must be non-negative")
        if self.minimum_subtype_count < 0.0:
            raise ValueError("minimum_subtype_count must be non-negative")
        if self.recovery_weight <= 0.0 or self.hard_negative_weight <= 0.0:
            raise ValueError("hard-negative weights must be positive")
        self.vocab = vocab

        self.visual_projection = nn.Linear(visual_dim, model_dim)
        self.hand_embedding = nn.Embedding(len(HANDS) + 1, model_dim)
        self.transition_embedding = nn.Embedding(len(TRANSITION_TYPES), model_dim)
        self.verb_embedding = nn.Embedding(vocab.size("verb"), model_dim)
        self.part_embedding = nn.Embedding(vocab.size("part"), model_dim)
        self.tool_embedding = nn.Embedding(vocab.size("tool"), model_dim)
        self.elapsed_projection = nn.Sequential(
            nn.Linear(1, model_dim),
            nn.GELU(),
            nn.Linear(model_dim, model_dim),
        )
        self.source_embedding = nn.Embedding(3, model_dim)  # worker, expert, SOP
        self.token_norm = nn.LayerNorm(model_dim)
        self.bos = nn.Parameter(torch.zeros(model_dim))
        self.missing_hand_tokens = nn.Parameter(torch.zeros(len(HANDS), model_dim))
        role_terms = 4 if self.role_fusion_interactions else 2
        self.role_fusion = nn.Sequential(
            nn.LayerNorm(role_terms * model_dim),
            nn.Linear(role_terms * model_dim, model_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(model_dim, model_dim),
            nn.LayerNorm(model_dim),
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=num_heads,
            dim_feedforward=4 * model_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.history_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )
        memory_layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=num_heads,
            dim_feedforward=4 * model_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.memory_encoder = nn.TransformerEncoder(memory_layer, num_layers=num_layers)
        self.expected_attention = nn.MultiheadAttention(
            model_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.observed_attention = nn.MultiheadAttention(
            model_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.expected_norm = nn.LayerNorm(model_dim)
        if self.use_effect_evidence:
            # Predictive-coding head: the causal history and normative context
            # predict the observed transition state; its residual is one more
            # evidence coordinate, trained on normal events only.
            self.effect_head = nn.Linear(model_dim, model_dim)

        self.mark_heads = nn.ModuleDict(
            {
                "hand": nn.Linear(model_dim, len(HANDS)),
                "transition": nn.Linear(model_dim, len(TRANSITION_TYPES)),
                "verb": nn.Linear(model_dim, vocab.size("verb")),
                "part": nn.Linear(model_dim, vocab.size("part")),
                "tool": nn.Linear(model_dim, vocab.size("tool")),
            }
        )
        time_outputs = (
            len(HANDS)
            if self.timing_clock == "competing"
            else 1 if self.timing_clock == "exponential" else 2
        )
        self.time_head = nn.Linear(model_dim, time_outputs)
        anomaly_input = (
            model_dim
            + len(self.EVIDENCE_NAMES)
            + (1 if self.use_effect_evidence else 0)
        )
        # The anomaly-head shape differs across controlled variants.  Isolate
        # its parameter initialization so it cannot change the random stream
        # used later by the shared backbone's training-time dropout.
        anomaly_rng_state = torch.random.get_rng_state()
        self.anomaly_head = nn.ModuleDict(
            {
                "encoder": nn.Sequential(
                    nn.LayerNorm(anomaly_input),
                    nn.Linear(anomaly_input, self.anomaly_hidden_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                ),
                # With an explicit recovery state the mutually exclusive
                # execution states are ordinary-normal, error, and recovery.
                # Deployment anomaly probability is still exactly P(error).
                "binary": nn.Linear(
                    self.anomaly_hidden_dim,
                    (
                        2
                        if self.factorized_recovery_state
                        else 3
                        if self.use_recovery_state
                        else 2
                    ),
                ),
                "subtype": nn.Linear(
                    self.anomaly_hidden_dim, len(ANOMALY_TYPES) - 1
                ),
            }
        )
        if self.factorized_recovery_state:
            self.anomaly_head["conditional_recovery"] = nn.Linear(
                self.anomaly_hidden_dim, 2
            )
        torch.random.set_rng_state(anomaly_rng_state)
        # Initialize shared parameters before constructing optional modules so
        # seed-matched controls retain bit-identical common weights.
        nn.init.normal_(self.bos, std=0.02)
        nn.init.normal_(self.missing_hand_tokens, std=0.02)
        if self.use_correction_state:
            # Optional-module initialization must not perturb the random stream
            # used by the shared HACT backbone during training.
            shared_rng_state = torch.random.get_rng_state()
            # The anomaly stream maintains one accepted execution state per
            # hand.  A transition is scored before it can update that state;
            # its predicted normal probability is the update posterior.  This
            # prevents an error from deterministically contaminating every
            # later event while retaining a differentiable, label-free update
            # at inference time.
            self.anomaly_head["state_init"] = nn.Sequential(
                nn.LayerNorm(model_dim),
                nn.Linear(model_dim, model_dim),
                nn.Tanh(),
            )
            if self.correction_state_mode == "joint":
                self.anomaly_head["state_fusion"] = nn.Sequential(
                    nn.LayerNorm(4 * model_dim),
                    nn.Linear(4 * model_dim, model_dim),
                    nn.GELU(),
                    nn.Linear(model_dim, model_dim),
                )
            self.anomaly_head["state_residual_norm"] = nn.LayerNorm(model_dim)
            self.anomaly_head["state_update"] = nn.GRUCell(model_dim, model_dim)
            if self.correction_use_state_belief:
                # Accepted and unresolved-error memories are complementary.
                # The former represents the trusted execution path; the
                # latter retains the content of a suspected error until a
                # recovery posterior clears it.  Both keep fixed left/right
                # slots and are updated only after a simultaneous group has
                # been scored.
                if self.correction_state_mode == "joint":
                    self.anomaly_head["state_error_fusion"] = nn.Sequential(
                        nn.LayerNorm(4 * model_dim),
                        nn.Linear(4 * model_dim, model_dim),
                        nn.GELU(),
                        nn.Linear(model_dim, model_dim),
                    )
                self.anomaly_head["state_error_residual_norm"] = nn.LayerNorm(
                    model_dim
                )
                self.anomaly_head["state_error_update"] = nn.GRUCell(
                    model_dim, model_dim
                )
            self.anomaly_head["state_adapter"] = nn.Sequential(
                nn.Linear(
                    model_dim
                    + (
                        model_dim
                        if self.correction_use_state_belief
                        else 0
                    )
                    + (len(HANDS) if self.correction_use_error_trace else 0)
                    + (
                        len(HANDS) * 3
                        if self.correction_use_state_belief
                        else 0
                    ),
                    self.anomaly_hidden_dim,
                ),
                nn.GELU(),
                nn.Linear(
                    self.anomaly_hidden_dim,
                    3 if self.use_recovery_state else 2,
                ),
            )
            nn.init.zeros_(self.anomaly_head["state_adapter"][-1].weight)
            nn.init.zeros_(self.anomaly_head["state_adapter"][-1].bias)
            torch.random.set_rng_state(shared_rng_state)
        if self.use_recovery_gate:
            # A zero log-odds ratio is an exact identity transformation of the
            # binary anomaly head, so staged recovery learning can never make
            # the starting checkpoint worse by construction.
            shared_rng_state = torch.random.get_rng_state()
            self.anomaly_head["recovery_encoder"] = nn.Sequential(
                nn.LayerNorm(self.anomaly_hidden_dim),
                nn.Linear(self.anomaly_hidden_dim, 2),
            )
            nn.init.zeros_(self.anomaly_head["recovery_encoder"][-1].weight)
            nn.init.zeros_(self.anomaly_head["recovery_encoder"][-1].bias)
            torch.random.set_rng_state(shared_rng_state)
        self.register_buffer(
            "supported_anomaly_types",
            torch.ones(len(ANOMALY_TYPES) - 1, dtype=torch.bool),
        )

        # Optional modules are created only when enabled, so checkpoints of
        # configurations without them load strictly.
        if self.use_bimanual_context and self.cross_hand_fusion in {
            "residual",
            "gated",
            "evidence",
        }:
            self.cross_bos = nn.Parameter(torch.zeros(model_dim))
            self.cross_attention = nn.MultiheadAttention(
                model_dim, num_heads, dropout=dropout, batch_first=True
            )
            self.cross_elapsed_projection = nn.Sequential(
                nn.Linear(1, model_dim),
                nn.GELU(),
                nn.Linear(model_dim, model_dim),
            )
            self.cross_gate = nn.Sequential(
                nn.LayerNorm(5 * model_dim),
                nn.Linear(5 * model_dim, model_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(model_dim, 1),
            )
            self.cross_dropout = nn.Dropout(self.cross_hand_dropout)
            self.cross_norm = nn.LayerNorm(model_dim)
            nn.init.normal_(self.cross_bos, std=0.02)
            nn.init.zeros_(self.cross_gate[-1].weight)
            nn.init.constant_(self.cross_gate[-1].bias, self.cross_gate_bias)
            if self.cross_hand_fusion == "evidence":
                self.cross_mark_heads = nn.ModuleDict(
                    {
                        name: nn.Linear(model_dim, self.mark_heads[name].out_features)
                        for name in ("transition", "verb", "part", "tool")
                    }
                )
                self.cross_time_head = nn.Linear(model_dim, 2)
                self.cross_anomaly_adapter = nn.Sequential(
                    nn.LayerNorm(len(self.COORDINATION_EVIDENCE_NAMES)),
                    nn.Linear(
                        len(self.COORDINATION_EVIDENCE_NAMES),
                        self.anomaly_hidden_dim,
                    ),
                )
                nn.init.zeros_(self.cross_anomaly_adapter[-1].weight)
                nn.init.zeros_(self.cross_anomaly_adapter[-1].bias)

    @property
    def uses_joint_timeline(self) -> bool:
        return self.use_bimanual_context and self.cross_hand_fusion == "joint"

    def _timing_values(
        self, batch: TransitionBatch
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return the event interval and support mask for the temporal factor.

        ``context`` preserves the legacy clock associated with the history
        stream. ``hand`` keeps role-aware joint mark context but factorizes
        timing over the current hand's asynchronous event process.
        """

        if self.timing_clock == "hand" or not self.uses_joint_timeline:
            return batch.delta_hand, batch.has_prev_hand
        return batch.delta_global, batch.has_prev_global

    def _embed_batch(self, batch: TransitionBatch, source: int) -> torch.Tensor:
        if self.use_time_likelihood:
            delta, has_previous = self._timing_values(batch)
            elapsed = torch.log1p(delta).unsqueeze(1) * has_previous.to(
                delta.dtype
            ).unsqueeze(1)
            elapsed_token = self.elapsed_projection(elapsed)
        else:
            elapsed_token = torch.zeros(
                (len(batch), self.model_dim), device=batch.visual.device
            )
        hand_token = (
            self.hand_embedding(batch.hand)
            if self.use_hand_identity
            else torch.zeros_like(elapsed_token)
        )
        semantic_token = (
            self.transition_embedding(batch.transition)
            + self.verb_embedding(batch.verb)
            + self.part_embedding(batch.part)
            + self.tool_embedding(batch.tool)
            if self.use_event_semantics
            else torch.zeros_like(elapsed_token)
        )
        tokens = (
            self.visual_projection(batch.visual)
            + hand_token
            + semantic_token
            + elapsed_token
            + self.source_embedding.weight[source]
        )
        return self.token_norm(tokens)

    def _embed_sop(
        self, marks: Sequence[Mapping[str, int]], device: torch.device
    ) -> torch.Tensor:
        if not marks:
            return torch.empty((0, self.model_dim), device=device)
        ordered_keys = sorted(
            {(mark["group_index"], mark["step_index"]) for mark in marks}
        )
        grouped = [
            [mark for mark in marks if (mark["group_index"], mark["step_index"]) == key]
            for key in ordered_keys
        ]
        flat = [mark for alternatives in grouped for mark in alternatives]
        verb = torch.tensor(
            [mark["verb"] for mark in flat], dtype=torch.long, device=device
        )
        part = torch.tensor(
            [mark["part"] for mark in flat], dtype=torch.long, device=device
        )
        tool = torch.tensor(
            [mark["tool"] for mark in flat], dtype=torch.long, device=device
        )
        hand = torch.full_like(verb, len(HANDS))
        transition = torch.full_like(verb, TRANSITION_TYPES.index("onset"))
        visual = torch.zeros((len(flat), self.model_dim), device=device)
        alternative_tokens = (
            visual
            + self.hand_embedding(hand)
            + self.transition_embedding(transition)
            + self.verb_embedding(verb)
            + self.part_embedding(part)
            + self.tool_embedding(tool)
            + self.source_embedding.weight[2]
        )
        step_tokens = []
        offset = 0
        for alternatives in grouped:
            step_tokens.append(
                alternative_tokens[offset : offset + len(alternatives)].mean(dim=0)
            )
            offset += len(alternatives)
        return self.token_norm(torch.stack(step_tokens))

    @staticmethod
    def _group_tokens(
        tokens: torch.Tensor, frames: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if len(tokens) != len(frames) or not len(tokens):
            raise ValueError(
                "Time-group tokens and frames must be non-empty and aligned"
            )
        if len(frames) > 1 and torch.any(frames[1:] < frames[:-1]):
            raise ValueError("Transition frames must be sorted before time grouping")
        unique_frames, inverse = torch.unique_consecutive(frames, return_inverse=True)
        groups = torch.zeros(
            (len(unique_frames), tokens.shape[-1]),
            dtype=tokens.dtype,
            device=tokens.device,
        )
        groups.index_add_(0, inverse, tokens)
        counts = torch.bincount(inverse, minlength=len(unique_frames)).to(tokens.dtype)
        return groups / counts.unsqueeze(1).clamp_min(1.0), inverse

    def _role_group_tokens(
        self, tokens: torch.Tensor, frames: torch.Tensor, hands: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode ordered left/right slots without averaging away hand roles."""

        if len(tokens) != len(frames) or len(tokens) != len(hands) or not len(tokens):
            raise ValueError("Role-group tokens, frames and hands must align")
        if len(frames) > 1 and torch.any(frames[1:] < frames[:-1]):
            raise ValueError("Transition frames must be sorted before time grouping")
        unique_frames, inverse = torch.unique_consecutive(frames, return_inverse=True)
        slot_index = inverse * len(HANDS) + hands
        flat_slots = torch.zeros(
            (len(unique_frames) * len(HANDS), tokens.shape[-1]),
            dtype=tokens.dtype,
            device=tokens.device,
        )
        flat_slots.index_add_(0, slot_index, tokens)
        counts = torch.bincount(
            slot_index, minlength=len(unique_frames) * len(HANDS)
        ).reshape(len(unique_frames), len(HANDS))
        slots = flat_slots.reshape(len(unique_frames), len(HANDS), -1)
        slots = slots / counts.to(tokens.dtype).unsqueeze(-1).clamp_min(1.0)
        missing = self.missing_hand_tokens.unsqueeze(0).expand_as(slots)
        slots = torch.where(counts.unsqueeze(-1) > 0, slots, missing)
        left, right = slots[:, 0], slots[:, 1]
        pair = (
            torch.cat((left, right, left - right, left * right), dim=-1)
            if self.role_fusion_interactions
            else torch.cat((left, right), dim=-1)
        )
        return self.role_fusion(pair), inverse

    def _global_groups(
        self, tokens: torch.Tensor, batch: TransitionBatch
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.use_role_fusion and self.use_hand_identity:
            return self._role_group_tokens(tokens, batch.frames, batch.hand)
        return self._group_tokens(tokens, batch.frames)

    def _encode_groups(
        self,
        tokens: torch.Tensor,
        frames: torch.Tensor,
        hands: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if hands is not None and self.use_role_fusion and self.use_hand_identity:
            groups, inverse = self._role_group_tokens(tokens, frames, hands)
        else:
            groups, inverse = self._group_tokens(tokens, frames)
        sequence = torch.cat((self.bos.unsqueeze(0), groups), dim=0)
        sequence = sequence + _sinusoidal_positions(
            len(sequence), self.model_dim, sequence.device
        )
        causal_mask = torch.triu(
            torch.ones(
                (len(sequence), len(sequence)), dtype=torch.bool, device=sequence.device
            ),
            diagonal=1,
        )
        encoded = self.history_encoder(sequence.unsqueeze(0), mask=causal_mask).squeeze(
            0
        )
        # State at position g is the strict history before time group g.
        contexts = encoded[:-1].index_select(0, inverse)
        return contexts, inverse

    def _per_hand_history_context(
        self, tokens: torch.Tensor, batch: TransitionBatch
    ) -> torch.Tensor:
        contexts = torch.empty_like(tokens)
        for hand_index in range(len(HANDS)):
            indices = torch.nonzero(batch.hand == hand_index, as_tuple=False).flatten()
            if not len(indices):
                continue
            hand_context, _ = self._encode_groups(
                tokens.index_select(0, indices), batch.frames.index_select(0, indices)
            )
            contexts.index_copy_(0, indices, hand_context)
        return contexts

    def _history_context(
        self, tokens: torch.Tensor, batch: TransitionBatch
    ) -> torch.Tensor:
        if not self.use_history:
            return self.bos.unsqueeze(0).expand(len(batch), -1)
        if self.uses_joint_timeline:
            contexts, _ = self._encode_groups(tokens, batch.frames, batch.hand)
            return contexts
        return self._per_hand_history_context(tokens, batch)

    def _cross_hand_context(
        self,
        tokens: torch.Tensor,
        self_history: torch.Tensor,
        batch: TransitionBatch,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Attend only to strictly earlier events from the opposite hand."""

        cross_keys = torch.cat((self.cross_bos.unsqueeze(0), tokens), dim=0)
        cross_keys = cross_keys + _sinusoidal_positions(
            len(cross_keys), self.model_dim, cross_keys.device
        )
        valid = (batch.hand.unsqueeze(1) != batch.hand.unsqueeze(0)) & (
            batch.frames.unsqueeze(1) > batch.frames.unsqueeze(0)
        )
        mask = torch.ones(
            (len(batch), len(batch) + 1), dtype=torch.bool, device=tokens.device
        )
        mask[:, 0] = False
        mask[:, 1:] = ~valid
        attended, _ = self.cross_attention(
            self_history.unsqueeze(0),
            cross_keys.unsqueeze(0),
            cross_keys.unsqueeze(0),
            attn_mask=mask,
            need_weights=False,
        )
        attended = attended.squeeze(0)
        available = valid.any(dim=1) & batch.has_prev_cross
        attended = attended * available.to(attended.dtype).unsqueeze(1)
        attended = self.cross_dropout(attended)

        if self.cross_hand_fusion == "gated":
            elapsed = self.cross_elapsed_projection(
                torch.log1p(batch.delta_cross).unsqueeze(1)
            )
            gate_input = torch.cat(
                (
                    self_history,
                    attended,
                    torch.abs(self_history - attended),
                    tokens,
                    elapsed,
                ),
                dim=-1,
            )
            gate = torch.sigmoid(self.cross_gate(gate_input))
        else:
            gate = torch.ones(
                (len(batch), 1), dtype=tokens.dtype, device=tokens.device
            )
        gate = gate * available.to(gate.dtype).unsqueeze(1)
        # A closed gate is an exact per-hand fallback.  Normalizing the sum
        # would perturb the self-history even when no cross-hand evidence is
        # available or the learned gate is near zero.
        fused = self_history + gate * self.cross_norm(attended)
        return fused, gate.squeeze(1), available, attended

    def build_memory(
        self,
        expert_batches: Sequence[TransitionBatch],
        sop_marks: Sequence[Mapping[str, int]],
    ) -> Mapping[str, Optional[torch.Tensor]]:
        expert_chunks = []
        keep_expert = self.use_expert_memory and (
            not self.training
            or self.memory_dropout == 0.0
            or bool(torch.rand((), device=self.bos.device) >= self.memory_dropout)
        )
        keep_sop = self.use_sop_memory and (
            not self.training
            or self.memory_dropout == 0.0
            or bool(torch.rand((), device=self.bos.device) >= self.memory_dropout)
        )
        if keep_expert:
            for batch in expert_batches:
                tokens = self._embed_batch(batch, source=1)
                groups, inverse = self._global_groups(tokens, batch)
                positions = _sinusoidal_positions(
                    len(groups), self.model_dim, tokens.device
                ).index_select(0, inverse)
                if self.use_role_fusion and self.use_hand_identity:
                    tokens = self.token_norm(tokens + groups.index_select(0, inverse))
                tokens = tokens + positions
                expert_chunks.append(
                    self.memory_encoder(tokens.unsqueeze(0)).squeeze(0)
                )
        expert_memory = torch.cat(expert_chunks, dim=0) if expert_chunks else None
        expected_chunks = [expert_memory] if expert_memory is not None else []
        if keep_sop and sop_marks:
            sop = self._embed_sop(sop_marks, self.bos.device)
            sop = sop + _sinusoidal_positions(len(sop), self.model_dim, sop.device)
            expected_chunks.append(self.memory_encoder(sop.unsqueeze(0)).squeeze(0))
        return {
            "expected": torch.cat(expected_chunks, dim=0) if expected_chunks else None,
            "visual": expert_memory,
        }

    @staticmethod
    def _attend(
        query: torch.Tensor,
        memory: Optional[torch.Tensor],
        attention: nn.MultiheadAttention,
    ) -> torch.Tensor:
        if memory is None or not len(memory):
            return torch.zeros_like(query)
        attended, _ = attention(
            query.unsqueeze(0),
            memory.unsqueeze(0),
            memory.unsqueeze(0),
            need_weights=False,
        )
        return attended.squeeze(0)

    @staticmethod
    def _categorical_surprise(
        logits: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        return -F.log_softmax(logits, dim=-1).gather(1, target.unsqueeze(1)).squeeze(1)

    def set_anomaly_support(self, class_counts: torch.Tensor) -> None:
        """Mask anomaly subtypes absent from this fold's training partition."""

        if class_counts.numel() not in {
            len(ANOMALY_TYPES),
            len(ANOMALY_TYPES) + 1,
        }:
            raise ValueError(
                f"Expected {len(ANOMALY_TYPES)} anomaly counts with an optional "
                f"recovery count, got {class_counts.numel()}"
            )
        supported = (
            class_counts.detach().to(self.supported_anomaly_types.device)[
                1 : len(ANOMALY_TYPES)
            ]
            >= self.minimum_subtype_count
        )
        if not torch.any(supported):
            raise ValueError("The training partition contains no anomalous events")
        self.supported_anomaly_types.copy_(supported)

    def _masked_type_logits(self, logits: torch.Tensor) -> torch.Tensor:
        minimum = torch.finfo(logits.dtype).min
        return logits.masked_fill(~self.supported_anomaly_types.unsqueeze(0), minimum)

    def hierarchical_log_probabilities(
        self, binary_logits: torch.Tensor, type_logits: torch.Tensor
    ) -> torch.Tensor:
        """Return log p(non-error) and log p(error type) from state heads."""

        binary_log_probability = F.log_softmax(binary_logits, dim=-1)
        type_log_probability = F.log_softmax(
            self._masked_type_logits(type_logits), dim=-1
        )
        error_log_probability = binary_log_probability[:, 1:2]
        if binary_logits.shape[-1] == 3:
            non_error_log_probability = torch.logsumexp(
                binary_log_probability[:, (0, 2)], dim=-1, keepdim=True
            )
        elif binary_logits.shape[-1] == 2:
            non_error_log_probability = binary_log_probability[:, :1]
        else:
            raise ValueError("Execution-state head must have two or three classes")
        return torch.cat(
            (
                non_error_log_probability,
                error_log_probability + type_log_probability,
            ),
            dim=-1,
        )

    @staticmethod
    def _time_terms(
        parameters: torch.Tensor, delta: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        mu = parameters[:, 0]
        sigma = F.softplus(parameters[:, 1]) + 1e-4
        log_delta = delta.clamp_min(1e-6).log()
        z = (log_delta - mu) / sigma
        nll = log_delta + sigma.log() + 0.5 * math.log(2.0 * math.pi) + 0.5 * z.square()
        survival = (0.5 * torch.erfc(z / math.sqrt(2.0))).clamp_min(1e-8)
        return nll, -survival.log()

    def _correction_state_anomaly(
        self,
        worker_tokens: torch.Tensor,
        base_binary_logits: torch.Tensor,
        batch: TransitionBatch,
        memory: Mapping[str, Optional[torch.Tensor]],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Score transitions with a causal, role-preserving accepted state.

        Events sharing a timestamp are scored from the same pre-event state.
        Their order therefore cannot leak into one another.  Each hand then
        updates once with its permutation-invariant set mean, softly gated by
        the posterior probability that the set is normal.
        """

        expected_memory = memory.get("expected")
        anchor = (
            expected_memory.mean(dim=0)
            if expected_memory is not None and len(expected_memory)
            else self.bos
        )
        initial = self.anomaly_head["state_init"](anchor)
        states = [initial, initial]
        error_states = [torch.zeros_like(initial), torch.zeros_like(initial)]
        error_trace = [initial.new_zeros(()), initial.new_zeros(())]
        state_beliefs = [
            initial.new_tensor([1.0, 0.0, 0.0]),
            initial.new_tensor([1.0, 0.0, 0.0]),
        ]
        binary_chunks = []
        gate_chunks = []

        _, group_counts = torch.unique_consecutive(
            batch.frames, return_counts=True
        )
        offset = 0
        for count_tensor in group_counts:
            count = int(count_tensor.item())
            stop = offset + count
            hands = batch.hand[offset:stop]
            own = torch.stack([states[int(value)] for value in hands.tolist()])
            if self.correction_state_mode == "joint":
                other = torch.stack(
                    [states[1 - int(value)] for value in hands.tolist()]
                )
                pair = torch.cat((own, other, own - other, own * other), dim=-1)
                accepted_context = self.anomaly_head["state_fusion"](pair)
            else:
                accepted_context = own
            state_residual = self.anomaly_head["state_residual_norm"](
                worker_tokens[offset:stop] - accepted_context
            )
            adapter_input = state_residual
            if self.correction_use_error_trace:
                trace = torch.stack(
                    [
                        torch.stack(
                            (error_trace[int(value)], error_trace[1 - int(value)])
                        )
                        for value in hands.tolist()
                    ]
                )
                adapter_input = torch.cat((adapter_input, trace), dim=-1)
            if self.correction_use_state_belief:
                own_error = torch.stack(
                    [error_states[int(value)] for value in hands.tolist()]
                )
                if self.correction_state_mode == "joint":
                    other_error = torch.stack(
                        [
                            error_states[1 - int(value)]
                            for value in hands.tolist()
                        ]
                    )
                    error_pair = torch.cat(
                        (
                            own_error,
                            other_error,
                            own_error - other_error,
                            own_error * other_error,
                        ),
                        dim=-1,
                    )
                    error_context = self.anomaly_head["state_error_fusion"](
                        error_pair
                    )
                else:
                    error_context = own_error
                error_residual = self.anomaly_head[
                    "state_error_residual_norm"
                ](worker_tokens[offset:stop] - error_context)
                # The fixed [left, right] ordering is identical for every
                # event.  The full adapter input exposes both what has been
                # accepted and what remains unresolved, without hand-crafted
                # recovery features or an external alarm rule.
                belief = torch.cat(state_beliefs, dim=-1).unsqueeze(0)
                belief = belief.expand(len(adapter_input), -1)
                adapter_input = torch.cat(
                    (adapter_input, error_residual, belief), dim=-1
                )
            binary = base_binary_logits[offset:stop] + self.anomaly_head[
                "state_adapter"
            ](
                adapter_input
            )
            # Both ordinary execution and an accepted corrective action may
            # advance memory.  Only the error state blocks the update.
            normal_probability = 1.0 - F.softmax(binary, dim=-1)[:, 1]

            binary_chunks.append(binary)
            gate_chunks.append(normal_probability)

            # Simultaneous events see the same state and update only after all
            # of them have been scored.  Multiple transitions from one hand at
            # a timestamp form a set rather than an arbitrary sequence.
            next_states = list(states)
            next_error_states = list(error_states)
            next_error_trace = list(error_trace)
            next_state_beliefs = list(state_beliefs)
            for hand_index in range(len(HANDS)):
                mask = hands == hand_index
                if not torch.any(mask):
                    continue
                token = worker_tokens[offset:stop][mask].mean(dim=0)
                gate = normal_probability[mask].mean()
                if not self.gate_execution_state:
                    gate = torch.ones_like(gate)
                candidate = self.anomaly_head["state_update"](
                    token, states[hand_index]
                )
                next_states[hand_index] = (
                    gate * candidate + (1.0 - gate) * states[hand_index]
                )
                next_error_trace[hand_index] = 1.0 - gate
                posterior = F.softmax(binary[mask], dim=-1).mean(dim=0)
                next_state_beliefs[hand_index] = posterior
                if self.correction_use_state_belief:
                    error_candidate = self.anomaly_head["state_error_update"](
                        token, error_states[hand_index]
                    )
                    # N retains a still-unresolved error, E writes the current
                    # suspicious event, and R clears it.  The coefficients are
                    # the model posterior itself; there is no external gate or
                    # persistence hyperparameter.
                    next_error_states[hand_index] = (
                        posterior[0] * error_states[hand_index]
                        + posterior[1] * error_candidate
                    )
            states = next_states
            error_states = next_error_states
            error_trace = next_error_trace
            state_beliefs = next_state_beliefs
            offset = stop

        return torch.cat(binary_chunks, dim=0), torch.cat(gate_chunks, dim=0)

    def forward(
        self,
        batch: TransitionBatch,
        expert_batches: Sequence[TransitionBatch] = (),
        sop_marks: Sequence[Mapping[str, int]] = (),
        memory: Optional[Mapping[str, Optional[torch.Tensor]]] = None,
    ) -> dict:
        worker_tokens = self._embed_batch(batch, source=0)
        history = self._history_context(worker_tokens, batch)
        cross_gate = torch.zeros(
            len(batch), dtype=worker_tokens.dtype, device=worker_tokens.device
        )
        cross_available = torch.zeros(
            len(batch), dtype=torch.bool, device=worker_tokens.device
        )
        coordination_evidence = None
        cross_normative_nll = None
        if self.use_bimanual_context and self.cross_hand_fusion in {
            "residual",
            "gated",
            "evidence",
        }:
            self_history = history
            cross_tokens = (
                worker_tokens.detach()
                if self.cross_hand_fusion == "evidence"
                else worker_tokens
            )
            cross_query = (
                self_history.detach()
                if self.cross_hand_fusion == "evidence"
                else self_history
            )
            fused_history, cross_gate, cross_available, cross_context = (
                self._cross_hand_context(cross_tokens, cross_query, batch)
            )
            if self.cross_hand_fusion == "evidence":
                # The main likelihood remains exactly per-hand.  A detached,
                # separately supervised cross stream contributes only anomaly
                # evidence, preventing it from explaining anomalies away.
                history = self_history
                cross_expected = self.cross_norm(cross_context)
                cross_logits = {
                    name: head(cross_expected)
                    for name, head in self.cross_mark_heads.items()
                }
                cross_surprises = [
                    self._categorical_surprise(cross_logits[name], getattr(batch, name))
                    for name in ("transition", "verb", "part", "tool")
                ]
                cross_time_nll, _ = self._time_terms(
                    self.cross_time_head(cross_expected), batch.delta_cross
                )
                valid_cross = cross_available.to(cross_time_nll.dtype)
                coordination_evidence = torch.stack(
                    tuple(value * valid_cross for value in cross_surprises)
                    + (cross_time_nll * valid_cross,),
                    dim=-1,
                )
                cross_normative_nll = coordination_evidence.sum(dim=-1)
            else:
                history = fused_history
            observed_tokens = worker_tokens
        elif self.use_bimanual_context:
            synchronous_groups, inverse = self._global_groups(worker_tokens, batch)
            group_context = synchronous_groups.index_select(0, inverse)
            observed_tokens = (
                self.token_norm(worker_tokens + group_context)
                if self.use_role_fusion and self.use_hand_identity
                else group_context
            )
        else:
            observed_tokens = worker_tokens
        if memory is None:
            memory = self.build_memory(expert_batches, sop_marks)
        expected_reference = self._attend(
            history, memory["expected"], self.expected_attention
        )
        observed_reference = self._attend(
            observed_tokens, memory["visual"], self.observed_attention
        )
        expected = self.expected_norm(history + expected_reference)

        # Marks follow the causal, role-preserving bimanual history, whereas
        # asynchronous timing is predicted from the current hand's own past.
        # This realizes a single factorization without a fusion coefficient:
        # p(mark_h, dt_h | H_L, H_R) = p(mark_h | H_L, H_R) p(dt_h | H_h).
        time_expected = expected
        if self.use_time_likelihood and self.timing_clock == "hand" and self.uses_joint_timeline:
            local_history = self._per_hand_history_context(worker_tokens, batch)
            local_reference = self._attend(
                local_history, memory["expected"], self.expected_attention
            )
            time_expected = self.expected_norm(local_history + local_reference)

        time_parameters = self.time_head(time_expected)
        mark_logits = {name: head(expected) for name, head in self.mark_heads.items()}
        if self.use_time_likelihood and self.timing_clock == "competing":
            # Cause-specific intensities jointly define which hand acts and
            # when the next event occurs.  The hand mark is therefore tied to
            # the normalized intensities instead of being predicted twice.
            rates = F.softplus(time_parameters) + 1e-4
            mark_logits["hand"] = rates.log()
        target_by_name = {
            "hand": batch.hand,
            "transition": batch.transition,
            "verb": batch.verb,
            "part": batch.part,
            "tool": batch.tool,
        }
        mark_surprises = {
            name: self._categorical_surprise(mark_logits[name], target)
            for name, target in target_by_name.items()
        }
        if not self.use_hand_identity:
            mark_surprises["hand"] = torch.zeros_like(mark_surprises["hand"])
        if not self.use_event_semantics:
            for name in ("transition", "verb", "part", "tool"):
                mark_surprises[name] = torch.zeros_like(mark_surprises[name])
        delta, has_previous = self._timing_values(batch)
        if self.use_time_likelihood:
            if self.timing_clock == "competing":
                total_rate = rates.sum(dim=-1)
                integrated_hazard = total_rate * batch.delta_global
                time_nll = -total_rate.log() + integrated_hazard
                valid_time = batch.has_prev_global.to(time_nll.dtype)
                time_nll = time_nll * valid_time
                survival_surprise = integrated_hazard * valid_time
            elif self.timing_clock == "exponential":
                rate = F.softplus(time_parameters.squeeze(-1)) + 1e-4
                integrated_hazard = rate * delta
                time_nll = -rate.log() + integrated_hazard
                valid_time = has_previous.to(time_nll.dtype)
                time_nll = time_nll * valid_time
                survival_surprise = integrated_hazard * valid_time
            else:
                time_nll, survival_surprise = self._time_terms(
                    time_parameters, delta
                )
                valid_time = has_previous.to(time_nll.dtype)
                time_nll = time_nll * valid_time
                survival_surprise = survival_surprise * valid_time
        else:
            time_nll = torch.zeros(len(batch), device=worker_tokens.device)
            survival_surprise = torch.zeros_like(time_nll)
        visual_residual = 1.0 - F.cosine_similarity(
            observed_tokens, observed_reference, dim=-1
        )
        if memory["visual"] is None:
            visual_residual = torch.zeros_like(time_nll)

        effect_residual = None
        if self.use_effect_evidence:
            effect_residual = 1.0 - F.cosine_similarity(
                self.effect_head(expected), observed_tokens.detach(), dim=-1
            )
        evidence_columns = [
            mark_surprises["hand"],
            mark_surprises["transition"],
            mark_surprises["verb"],
            mark_surprises["part"],
            mark_surprises["tool"],
            time_nll,
            survival_surprise,
            visual_residual,
        ]
        if effect_residual is not None:
            evidence_columns.append(effect_residual)
        evidence = torch.stack(tuple(evidence_columns), dim=-1)
        comparison_residual = (
            observed_tokens - observed_reference
            if memory["visual"] is not None
            else observed_tokens
        )
        correction_gate = torch.ones_like(time_nll)
        anomaly_input = torch.cat((comparison_residual, evidence), dim=-1)
        anomaly_hidden = self.anomaly_head["encoder"](anomaly_input)
        if coordination_evidence is not None:
            anomaly_hidden = anomaly_hidden + self.cross_anomaly_adapter(
                coordination_evidence
            )
        base_error_logits = self.anomaly_head["binary"](anomaly_hidden)
        conditional_recovery_logits = None
        if self.factorized_recovery_state:
            conditional_recovery_logits = self.anomaly_head[
                "conditional_recovery"
            ](anomaly_hidden)
            error_log_probability = F.log_softmax(base_error_logits, dim=-1)
            recovery_log_probability = F.log_softmax(
                conditional_recovery_logits, dim=-1
            )
            binary_logits = torch.stack(
                (
                    error_log_probability[:, 0]
                    + recovery_log_probability[:, 0],
                    error_log_probability[:, 1],
                    error_log_probability[:, 0]
                    + recovery_log_probability[:, 1],
                ),
                dim=-1,
            )
        else:
            binary_logits = base_error_logits
        type_logits = self.anomaly_head["subtype"](anomaly_hidden)
        correction_trainable = any(
            parameter.requires_grad for parameter in self.anomaly_head.parameters()
        )
        if self.correction_enabled and correction_trainable:
            binary_logits, correction_gate = self._correction_state_anomaly(
                worker_tokens,
                binary_logits,
                batch,
                memory,
            )
        recovery_logits = None
        if self.use_recovery_gate:
            recovery_logits = self.anomaly_head["recovery_encoder"](anomaly_hidden)
            if self.recovery_gate_enabled:
                recovery_log_odds = recovery_logits[:, 1] - recovery_logits[:, 0]
                binary_logits = torch.stack(
                    (binary_logits[:, 0], binary_logits[:, 1] - recovery_log_odds),
                    dim=-1,
                )
        anomaly_logits = self.hierarchical_log_probabilities(binary_logits, type_logits)
        return {
            "mark_logits": mark_logits,
            "time_parameters": time_parameters,
            "time_nll": time_nll,
            "survival_surprise": survival_surprise,
            "visual_residual": visual_residual,
            "effect_residual": effect_residual,
            "evidence": evidence,
            "anomaly_logits": anomaly_logits,
            "binary_logits": binary_logits,
            "base_error_logits": base_error_logits,
            "conditional_recovery_logits": conditional_recovery_logits,
            "recovery_logits": recovery_logits,
            "type_logits": type_logits,
            "cross_gate": cross_gate,
            "cross_available": cross_available,
            "coordination_evidence": coordination_evidence,
            "cross_normative_nll": cross_normative_nll,
            "correction_gate": correction_gate,
            "memory": memory,
        }

    def normative_loss(
        self, outputs: Mapping[str, object], batch: TransitionBatch
    ) -> torch.Tensor:
        normal = batch.anomaly == 0
        if not torch.any(normal):
            return next(self.parameters()).sum() * 0.0
        weights = batch.event_weight * normal.to(batch.event_weight.dtype)
        joint_nll = torch.zeros(len(batch), device=batch.visual.device)
        targets = {
            "hand": batch.hand,
            "transition": batch.transition,
            "verb": batch.verb,
            "part": batch.part,
            "tool": batch.tool,
        }
        if not self.use_hand_identity:
            targets.pop("hand")
        if not self.use_event_semantics:
            for name in ("transition", "verb", "part", "tool"):
                targets.pop(name)
        mark_logits = outputs["mark_logits"]
        for name, target in targets.items():
            joint_nll = joint_nll + self._categorical_surprise(
                mark_logits[name], target
            )
        mark_loss = (joint_nll * weights).sum() / weights.sum().clamp_min(1e-8)
        loss = mark_loss
        if self.use_effect_evidence:
            effect_residual = outputs.get("effect_residual")
            if effect_residual is not None:
                effect_loss = (
                    effect_residual * weights
                ).sum() / weights.sum().clamp_min(1e-8)
                loss = loss + effect_loss
        if self.use_time_likelihood:
            _, has_previous = self._timing_values(batch)
            temporal_nll = outputs["time_nll"]
            time_weights = (
                batch.time_weight
                * normal.to(batch.time_weight.dtype)
                * has_previous.to(batch.time_weight.dtype)
            )
            if torch.any(time_weights > 0):
                time_loss = (
                    temporal_nll * time_weights
                ).sum() / time_weights.sum().clamp_min(1e-8)
                loss = loss + time_loss
        cross_nll = outputs.get("cross_normative_nll")
        if cross_nll is not None and self.cross_auxiliary_weight > 0.0:
            cross_weights = (
                batch.event_weight
                * normal.to(batch.event_weight.dtype)
                * outputs["cross_available"].to(batch.event_weight.dtype)
            )
            if torch.any(cross_weights > 0):
                cross_loss = (
                    cross_nll * cross_weights
                ).sum() / cross_weights.sum().clamp_min(1e-8)
                loss = loss + self.cross_auxiliary_weight * cross_loss
        return loss

    def anomaly_loss(
        self,
        outputs: Mapping[str, torch.Tensor],
        batch: TransitionBatch,
        class_counts: torch.Tensor,
        recovery_only: bool = False,
    ) -> torch.Tensor:
        anomaly_counts = class_counts[: len(ANOMALY_TYPES)]
        recovery_count = (
            class_counts[len(ANOMALY_TYPES)]
            if class_counts.numel() > len(ANOMALY_TYPES)
            else torch.zeros_like(class_counts[0])
        )
        if recovery_only:
            recovery_logits = outputs.get("recovery_logits")
            if recovery_logits is None:
                raise ValueError("Recovery-only loss requires use_recovery_gate=True")
            recovery_targets = batch.recovery.to(torch.long)
            recovery_counts = torch.stack(
                (
                    (anomaly_counts.sum() - recovery_count).clamp_min(0.0),
                    recovery_count,
                )
            )
            return balanced_softmax_loss(
                recovery_logits,
                recovery_targets,
                recovery_counts,
                batch.event_weight,
            )
        binary_targets = (batch.anomaly != 0).to(torch.long)
        if self.factorized_recovery_state:
            state_logits = outputs["binary_logits"]
            factorized_error_logits = torch.stack(
                (
                    torch.logsumexp(state_logits[:, (0, 2)], dim=-1),
                    state_logits[:, 1],
                ),
                dim=-1,
            )
            error_counts = torch.stack(
                (anomaly_counts[0], anomaly_counts[1:].sum())
            )
            error_loss = balanced_softmax_loss(
                factorized_error_logits,
                binary_targets,
                error_counts,
                batch.event_weight,
            )
            non_error = batch.anomaly == 0
            recovery_targets = batch.recovery[non_error].to(torch.long)
            recovery_counts = torch.stack(
                (
                    (anomaly_counts[0] - recovery_count).clamp_min(0.0),
                    recovery_count,
                )
            )
            recovery_loss = balanced_softmax_loss(
                state_logits[non_error][:, (0, 2)],
                recovery_targets,
                recovery_counts,
                batch.event_weight[non_error],
            )
            binary_loss = error_loss + recovery_loss
        elif self.use_recovery_state:
            binary_targets = binary_targets.clone()
            binary_targets[batch.recovery.to(torch.bool)] = 2
            binary_counts = torch.stack(
                (
                    (anomaly_counts[0] - recovery_count).clamp_min(0.0),
                    anomaly_counts[1:].sum(),
                    recovery_count,
                )
            )
        else:
            binary_counts = torch.stack(
                (anomaly_counts[0], anomaly_counts[1:].sum())
            )
        binary_weights = batch.event_weight.clone()
        if not self.use_recovery_state and self.recovery_weight != 1.0:
            binary_weights = binary_weights * torch.where(
                batch.recovery.to(torch.bool),
                torch.as_tensor(
                    self.recovery_weight,
                    dtype=binary_weights.dtype,
                    device=binary_weights.device,
                ),
                torch.ones((), dtype=binary_weights.dtype, device=binary_weights.device),
            )
        if self.hard_negative_weight != 1.0:
            event_rows = {}
            for index, record in enumerate(batch.records):
                event_rows.setdefault(
                    record.event_id,
                    (record.event_start_frame, record.hand, record.anomaly),
                )
            ordered = sorted(
                event_rows.items(), key=lambda item: (item[1][0], item[1][1], item[0])
            )
            hard_events = set()
            for index, (event_id, (_, _, anomaly)) in enumerate(ordered):
                if anomaly != 0:
                    continue
                neighbors = ordered[max(0, index - 1) : index] + ordered[
                    index + 1 : index + 2
                ]
                if any(values[2] != 0 for _, values in neighbors):
                    hard_events.add(event_id)
            hard_mask = torch.tensor(
                [record.event_id in hard_events for record in batch.records],
                dtype=torch.bool,
                device=binary_weights.device,
            )
            binary_weights = binary_weights * torch.where(
                hard_mask,
                torch.as_tensor(
                    self.hard_negative_weight,
                    dtype=binary_weights.dtype,
                    device=binary_weights.device,
                ),
                torch.ones((), dtype=binary_weights.dtype, device=binary_weights.device),
            )
        if not self.factorized_recovery_state:
            binary_loss = balanced_softmax_loss(
                outputs["binary_logits"], binary_targets, binary_counts, binary_weights
            )
        anomalous = batch.anomaly != 0
        supported_targets = torch.zeros_like(anomalous)
        if torch.any(anomalous):
            anomaly_indices = batch.anomaly[anomalous] - 1
            supported_targets[anomalous] = self.supported_anomaly_types.index_select(
                0, anomaly_indices
            )
        typed = anomalous & supported_targets
        if not torch.any(typed) or self.subtype_loss_weight == 0.0:
            return binary_loss
        type_logits = self._masked_type_logits(outputs["type_logits"][typed])
        type_loss = balanced_softmax_loss(
            type_logits,
            batch.anomaly[typed] - 1,
            anomaly_counts[1:],
            batch.event_weight[typed],
        )
        return binary_loss + self.subtype_loss_weight * type_loss

    def checkpoint_config(self) -> dict:
        return {
            "visual_dim": self.visual_dim,
            "model_dim": self.model_dim,
            "num_layers": self.num_layers,
            "num_heads": self.num_heads,
            "dropout": self.dropout_rate,
            "use_history": self.use_history,
            "use_bimanual_context": self.use_bimanual_context,
            "use_expert_memory": self.use_expert_memory,
            "use_sop_memory": self.use_sop_memory,
            "use_time_likelihood": self.use_time_likelihood,
            "timing_clock": self.timing_clock,
            "use_hand_identity": self.use_hand_identity,
            "use_event_semantics": self.use_event_semantics,
            "use_role_fusion": self.use_role_fusion,
            "role_fusion_interactions": self.role_fusion_interactions,
            "gate_execution_state": self.gate_execution_state,
            "use_recovery_state": self.use_recovery_state,
            "factorized_recovery_state": self.factorized_recovery_state,
            "use_recovery_gate": self.use_recovery_gate,
            "use_correction_state": self.use_correction_state,
            "correction_state_mode": self.correction_state_mode,
            "correction_use_error_trace": self.correction_use_error_trace,
            "correction_use_state_belief": self.correction_use_state_belief,
            "anomaly_hidden_dim": self.anomaly_hidden_dim,
            "subtype_loss_weight": self.subtype_loss_weight,
            "minimum_subtype_count": self.minimum_subtype_count,
            "recovery_weight": self.recovery_weight,
            "hard_negative_weight": self.hard_negative_weight,
            "memory_dropout": self.memory_dropout,
            "cross_hand_fusion": self.cross_hand_fusion,
            "cross_hand_dropout": self.cross_hand_dropout,
            "cross_gate_bias": self.cross_gate_bias,
            "cross_auxiliary_weight": self.cross_auxiliary_weight,
            "train_visual_during_anomaly": self.train_visual_during_anomaly,
            "use_effect_evidence": self.use_effect_evidence,
            "vocab": self.vocab.to_dict(),
        }
