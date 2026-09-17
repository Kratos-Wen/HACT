# HACT: Hand-Aware Transition Modeling for Bimanual Procedural Anomaly Detection

Official code of **"Hand-Aware Transition Modeling for Bimanual Procedural Anomaly Detection"**.

Di Wen\*, Jimmy Weissert\*, Luc Maria Scherrer\*, Cedric Zöllner\*, Kailun Yang, Ruiping Liu, Yufan Chen,
Jiale Wei, Junwei Zheng, Kunyu Peng† (\*equal contribution, †corresponding author)

HACT detects procedural anomalies in egocentric video of bimanual assembly. A frozen video encoder and a
bimanual event decoder predict per-hand events. A role-preserving history keeps the left and the right hand
in fixed slots, and a marked temporal point process over this history assigns every observed hand transition
a semantic and temporal surprisal. A supervised evidence head and a two-state filter turn the surprisals
into a per-hand anomaly posterior. No annotation of the query recording enters the score.

## Results

Fully predicted, participant-disjoint five-fold evaluation on IMPACT-ego Reassembly A (public). R-FPR is the
recovery false-positive rate at the threshold that reaches recall .3 on the validation participants.

| Method | AUPRC | F1 | R-FPR | R-FPR (covered) |
|---|---|---|---|---|
| Causal TCN | .095 | .224 | .284 | .284 |
| MistSense | .129 | .224 | .325 | .325 |
| EgoPER-V | .115 | .182 | .448 | .450 |
| **HACT** | **.160** | **.252** | **.225** | **.281** |

Transfer to Reassembly B, the Reassembly A fold models applied unchanged:

| Method | AUPRC | F1 |
|---|---|---|
| Causal TCN | .153 | .343 |
| MistSense | .218 | .347 |
| **HACT** | **.238** | **.358** |

## Installation

```bash
conda create -n hact python=3.10 -y && conda activate hact
pip install -r requirements.txt
```

Training one fold takes a few minutes on a single GPU; the model has 1.49M parameters beyond the frozen encoder.

## Data

HACT uses the public [IMPACT](https://huggingface.co/datasets/KratosWen/IMPACT) v1.1 release (CC BY-NC-SA 4.0),
egocentric view. This repository ships the converted event annotations, the participant-disjoint folds, and the
vocabulary, so only the features have to be prepared:

```bash
# VideoMAE V2 features shipped with IMPACT -> 15 fps cache read by HACT
python tools/build_feature_cache.py --impact-features <IMPACT>/features/VideoMAEv2 \
    --benchmark data/impact_public/reassembly_a --output data/features/impact_videomaev2_15fps
python tools/build_feature_cache.py --impact-features <IMPACT>/features/VideoMAEv2 \
    --benchmark data/impact_public/reassembly_b_transfer --output data/features/impact_videomaev2_15fps
```

| Path | Content |
|---|---|
| `data/impact_public/reassembly_a` | 41 executions by 12 participants, two reference executions, five folds |
| `data/impact_public/reassembly_b_transfer` | the same folds with the 10 Reassembly B recordings as test recordings |
| `configs/public_impact` | verb, part and tool vocabulary, and the step list derived from the reference executions |

The public release has no contact-onset annotation; the converted annotations place the onset at the midpoint
of each event (`data/impact_public/preparation_report.json`).

## Training and evaluation

All commands run from the repository root.

```bash
# HACT: five folds, rescoring on predicted events, two-state filter, evaluation
NAME=hact_reassembly_a GPUS="0 1 2 3" scripts/run_hact.sh 0

# dense baselines on the same features and folds
GPUS="0 1 2 3" scripts/run_frame_baseline.sh causal_tcn
GPUS="0 1 2 3" scripts/run_frame_baseline.sh mistsense

# transfer to Reassembly B without retraining
NAME=hact_reassembly_a scripts/run_transfer_b.sh 0
```

Results are written to `outputs/results/`: `<name>_seed<s>_frame.json` holds the pooled out-of-fold AUPRC and F1,
`<name>_seed<s>_recovery.json` the recovery false-positive rate at the validation-selected operating points
together with the achieved test recall and the rate on covered recovery frames.

Every threshold, the temperature, early stopping and model selection use the validation participants of the
fold; the test recordings are read once.

### Ablations

| Config | Variant |
|---|---|
| `hact_reassembly_a` | full model |
| `hact_reassembly_a_norole` | no role-preserving history |
| `hact_reassembly_a_nointeract` | no interaction terms between the two hand slots |
| `hact_reassembly_a_nomemory` | no execution memory |
| `hact_reassembly_a_gatedwrite` | memory write scaled by one minus the anomaly probability |
| `hact_reassembly_a_visualonly` | event decoder and visual residual only |
| `hact_reassembly_a_norecw` | no recovery-frame weight in the evidence loss |

```bash
NAME=hact_reassembly_a_norole GPUS="0 1 2 3" scripts/run_hact.sh 0
```

## Repository layout

| Path | Content |
|---|---|
| `pipeline/bimanual_transition_model.py` | event decoder, role-preserving transition model, evidence head, execution memory |
| `pipeline/bimanual_transition_data.py` | events, transitions, folds |
| `pipeline/step2_bimanual_transition.py` | staged training and inference (`run --config ... --fold ... --seed ...`) |
| `integrations/rescore_hact_transitions.py` | scoring on fully predicted events |
| `integrations/apply_hact_two_state_filter.py` | two-state anomaly filter |
| `integrations/evaluate_fully_predicted.py` | shared frame-level evaluation of all methods |
| `integrations/recovery_at_validation_recall.py` | recovery false-positive rate at validation-selected thresholds |
| `integrations/train_frame_baselines.py` | causal TCN and MistSense baselines |
| `integrations/zeroshot_vlm_baseline.py` | zero-shot Qwen2.5-VL baseline |
| `tools/build_feature_cache.py` | 15 fps feature cache from the IMPACT features |

The second bimanual collection reported in the paper will be added to this repository together with its
configurations once it is released.

## Citation

```bibtex
@article{wen2026hact,
  title   = {Hand-Aware Transition Modeling for Bimanual Procedural Anomaly Detection},
  author  = {Wen, Di and Weissert, Jimmy and Scherrer, Luc Maria and Z{\"o}llner, Cedric and Yang, Kailun and
             Liu, Ruiping and Chen, Yufan and Wei, Jiale and Zheng, Junwei and Peng, Kunyu},
  year    = {2026}
}
```

## License

The code is released under the MIT License. The annotations in `data/impact_public` are derived from IMPACT
and are distributed under CC BY-NC-SA 4.0.
