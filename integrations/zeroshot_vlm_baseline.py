#!/usr/bin/env python3
"""Zero-shot vision--language mistake detection as a reference baseline.

In the spirit of zero-shot procedural mistake detection with a vision--language
model (Ozsoy et al., 2026), the model receives the numbered step list of the
procedure, a short description of what it reported for the preceding clips,
and the frames of the current clip, and rates the probability that the worker
makes a procedural mistake in that clip.  Open 7B-scale models do not produce
reliable per-action segmentations of long bimanual recordings, so the clip
probability is held over the clip's frames as the frame score.  The model sees
raw RGB frames rather than the cached features, so this row is a different
information regime and is reported as a reference.

Outputs the dense layout read by ``evaluate_fully_predicted.py dense``: one
``fold_k`` directory per benchmark fold with validation/test score files that
share the same zero-shot scores.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate_fully_predicted import _read  # noqa: E402


def step_list(sop_path: Path) -> list[str]:
    sop = _read(sop_path)
    labels: list[str] = []
    for group in sop["steps"]:
        for step in group:
            label = step["step_label"]
            if isinstance(label, list):
                label = " or ".join(label)
            labels.append(str(label).replace(":", " ").replace("_", " "))
    return labels


def build_prompt(procedure: str, steps: list[str], history: list[str], clip_seconds: float) -> str:
    numbered = "\n".join(f"{i + 1}. {s}" for i, s in enumerate(steps))
    hist = "; ".join(history[-12:]) if history else "nothing yet"
    return (
        f"You monitor a worker performing the procedure '{procedure}' with both hands. "
        f"The correct steps of the procedure are:\n{numbered}\n"
        f"What you reported for the earlier parts of this video: {hist}.\n"
        f"The frames below cover the next {clip_seconds:.0f} seconds, sampled every two seconds. "
        "Describe in one sentence what the hands do in these frames. Then estimate the probability, between 0 and 1, "
        "that the worker makes a procedural mistake in this clip: a wrong tool, a wrong part, a step out of order, "
        "mishandling of a component, or wrong timing relative to the steps already completed. "
        'Answer only with JSON: {"description": "...", "mistake_probability": p}.'
    )


def parse_answer(text: str) -> tuple[str, float]:
    text = text.replace("```json", "").replace("```", "")
    start, end = text.find("{"), text.rfind("}")
    payload = None
    if start != -1 and end > start:
        try:
            payload = json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            payload = None
    description = ""
    probability = None
    if isinstance(payload, dict):
        description = str(payload.get("description", ""))[:120]
        try:
            probability = float(payload.get("mistake_probability"))
        except (TypeError, ValueError):
            probability = None
    if probability is None:
        match = re.search(r"mistake_probability\"?\s*[:=]\s*\"?([0-9]*\.?[0-9]+)", text)
        probability = float(match.group(1)) if match else 0.0
    return description, min(1.0, max(0.0, probability))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--video-root", type=Path, required=True, help="directory searched recursively for <video_id>.mp4")
    parser.add_argument("--sop", type=Path, required=True)
    parser.add_argument("--procedure-name", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--model", default="Qwen/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--clip-seconds", type=float, default=20.0)
    parser.add_argument("--sample-fps", type=float, default=0.5)
    parser.add_argument("--max-pixels", type=int, default=336 * 336)
    parser.add_argument("--feature-fps", type=float, default=15.0)
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=96)
    args = parser.parse_args()

    import cv2  # noqa: WPS433
    import torch
    from PIL import Image  # noqa: WPS433
    from transformers import AutoProcessor
    from transformers.models.qwen2_5_vl import Qwen2_5_VLForConditionalGeneration

    cv2.setNumThreads(2)

    manifest = _read(args.benchmark / "manifest.json")
    records = sorted(manifest["workers"], key=lambda r: str(r["video_id"]))
    records = records[args.shard :: args.num_shards]
    videos = {p.stem: p for p in args.video_root.rglob("*.mp4")}
    steps = step_list(args.sop)
    raw_dir = args.output_root / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map={"": 0}
    ).eval()
    processor = AutoProcessor.from_pretrained(args.model, min_pixels=128 * 128, max_pixels=args.max_pixels)
    torch.manual_seed(0)

    for record in records:
        video_id = str(record["video_id"])
        out_path = raw_dir / f"{video_id}.json"
        if out_path.exists():
            continue
        if video_id not in videos:
            raise FileNotFoundError(f"No video for {video_id} under {args.video_root}")
        capture = cv2.VideoCapture(str(videos[video_id]))
        native_fps = float(capture.get(cv2.CAP_PROP_FPS)) or 30.0
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frame_count / native_fps

        def read_frames(indices):
            frames = []
            for index in indices:
                capture.set(cv2.CAP_PROP_POS_FRAMES, int(index))
                ok, frame = capture.read()
                if ok:
                    frames.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
            return frames

        feature_length = int(record["feature_length"])
        scores = np.zeros(feature_length, dtype=np.float32)
        history: list[str] = []
        clips = []
        clip_start = 0.0
        while clip_start < duration:
            clip_end = min(duration, clip_start + args.clip_seconds)
            times = np.arange(clip_start, clip_end, 1.0 / args.sample_fps)
            indices = np.clip((times * native_fps).astype(int), 0, max(frame_count - 1, 0))
            images = read_frames(indices)
            if not images:
                clip_start = clip_end
                continue
            prompt = build_prompt(args.procedure_name, steps, history, clip_end - clip_start)
            messages = [{"role": "user", "content": [*({"type": "image"} for _ in images), {"type": "text", "text": prompt}]}]
            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = processor(text=[text], images=images, return_tensors="pt").to(model.device)
            with torch.no_grad():
                generated = model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=False)
            answer = processor.batch_decode(generated[:, inputs["input_ids"].shape[1] :], skip_special_tokens=True)[0]
            description, probability = parse_answer(answer)
            if description:
                history.append(description)
            start = int(round(clip_start * args.feature_fps))
            end = min(feature_length, int(round(clip_end * args.feature_fps)) + 1)
            if end > start:
                scores[start:end] = np.maximum(scores[start:end], probability)
            clips.append({"start": clip_start, "end": clip_end, "answer": answer, "description": description, "mistake_probability": probability})
            clip_start = clip_end
        capture.release()
        out_path.write_text(json.dumps({"video_id": video_id, "clips": clips, "scores": scores.tolist()}) + "\n")
        probs = [c["mistake_probability"] for c in clips] or [0.0]
        print(f"[done] {video_id}: {len(clips)} clips, mean p={np.mean(probs):.3f} max p={np.max(probs):.3f}", flush=True)

    if args.num_shards == 1 or args.shard == args.num_shards - 1:
        assemble(args.benchmark, args.output_root, args.model)


def assemble(benchmark: Path, output_root: Path, model_name: str) -> None:
    """Build the dense fold layout once every video has a raw result."""

    manifest = _read(benchmark / "manifest.json")
    splits = _read(benchmark / "splits.json")["folds"]
    raw_dir = output_root / "raw"
    scores = {}
    for record in manifest["workers"]:
        path = raw_dir / f"{record['video_id']}.json"
        if not path.exists():
            print(f"[assemble] missing {path.name}; run remaining shards first")
            return
        scores[str(record["video_id"])] = np.asarray(_read(path)["scores"], dtype=np.float32)
    for split in splits:
        fold_dir = output_root / f"fold_{int(split['fold'])}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(fold_dir / "validation_scores.npz", **{v: scores[v] for v in split["val"]})
        np.savez_compressed(fold_dir / "test_scores.npz", **{v: scores[v] for v in split["test"]})
        (fold_dir / "meta.json").write_text(
            json.dumps(
                {
                    "method": "zeroshot_vlm",
                    "model": model_name,
                    "fold": int(split["fold"]),
                    "validation_video_ids": list(split["val"]),
                    "test_video_ids": list(split["test"]),
                    "test_annotations_consumed_by_model": False,
                    "training": "none (zero-shot)",
                },
                indent=2,
            )
            + "\n"
        )
    print("[assemble] dense layout written")


if __name__ == "__main__":
    main()
