from __future__ import annotations

import json
from pathlib import Path

import torch


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    ckpt_dir = root / "checkpoints" / "slrt"
    ckpts = sorted(list(ckpt_dir.rglob("*.ckpt")) + list(ckpt_dir.rglob("*.pth")) + list(ckpt_dir.rglob("*.pt")))
    vocabs = sorted(list(ckpt_dir.rglob("*.vocab")) + list(ckpt_dir.rglob("*.json")))

    if not ckpts:
        print("No checkpoint found under checkpoints/slrt")
        return

    ckpt_path = ckpts[0]
    print("checkpoint:", ckpt_path)
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    state = checkpoint.get("model_state", checkpoint.get("state_dict", checkpoint))

    target_keys = [
        "recognition_network.visual_head.gloss_output_layer.weight",
        "recognition_network.visual_head_keypoint.gloss_output_layer.weight",
        "recognition_network.visual_head_fuse.gloss_output_layer.weight",
    ]
    class_count = None
    for key in target_keys:
        if key in state:
            class_count = int(state[key].shape[0])
            print("class_count_from", key, "=", class_count)
            break
    if class_count is None:
        for key, value in state.items():
            if key.endswith("gloss_output_layer.weight"):
                class_count = int(value.shape[0])
                print("class_count_from", key, "=", class_count)
                break

    if class_count is None:
        print("Could not infer class count from checkpoint keys")
        return

    if vocabs:
        vocab_path = vocabs[0]
        print("vocab:", vocab_path)
        with vocab_path.open("r", encoding="utf-8") as f:
            vocab = json.load(f)
        print("vocab_size:", len(vocab))
        if len(vocab) != class_count:
            print("MISMATCH: vocab size != checkpoint class count")
        else:
            print("OK: vocab size matches class count")
    else:
        print("No vocab file found; runtime will synthesize token labels")


if __name__ == "__main__":
    main()
