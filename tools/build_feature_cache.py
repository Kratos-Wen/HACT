#!/usr/bin/env python3
"""Build the 15 fps feature cache HACT reads from the VideoMAE V2 features shipped with IMPACT v1.1.

IMPACT ships one array per recording, ``features/VideoMAEv2/<video_id>.npy``, of shape
``(1408, T)`` with one column per native video frame.  HACT works at 15 frames per second:
a recording of ``T`` native frames yields ``ceil(T * 15 / fps)`` cache frames, and cache frame ``i`` is
the native frame ``min(round(i * fps / 15), T - 1)`` (round half to even).  The native frame rate of each recording is read from its parsed
annotation (``fps``).  The output is ``<video_id>_features.npz`` with ``features`` of shape
``(N, 1408)`` (float32) and ``frame_ids`` (int64, the native frame indices).

usage:
  python tools/build_feature_cache.py --impact-features <IMPACT>/features/VideoMAEv2 \
      --benchmark data/impact_public/reassembly_a --output data/features/impact_videomaev2_15fps
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

TARGET_FPS = 15.0


def frame_ids(native_frames: int, native_fps: float, target_fps: float = TARGET_FPS) -> np.ndarray:
    count = int(np.ceil(native_frames * target_fps / native_fps))
    ids = np.rint(np.arange(count, dtype=np.float64) * native_fps / target_fps).astype(np.int64)
    return np.minimum(ids, native_frames - 1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--impact-features", type=Path, required=True)
    parser.add_argument("--benchmark", type=Path, required=True, help="benchmark directory with manifest.json")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads((args.benchmark / "manifest.json").read_text())
    args.output.mkdir(parents=True, exist_ok=True)
    for record in manifest["workers"] + manifest["experts"]:
        video_id = str(record["video_id"])
        target = args.output / f"{video_id}_features.npz"
        if target.exists():
            continue
        native = np.load(args.impact_features / f"{video_id}.npy")  # (1408, T)
        fps = float(json.loads(Path(record["parsed_annotation_path"]).read_text())["fps"])
        ids = frame_ids(native.shape[1], fps)
        np.savez(target, features=np.ascontiguousarray(native[:, ids].T, dtype=np.float32), frame_ids=ids)
        print(f"{video_id}: {native.shape[1]} native frames at {fps:.3f} fps -> {len(ids)} frames")


if __name__ == "__main__":
    main()
