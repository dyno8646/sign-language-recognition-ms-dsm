from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)


@dataclass
class RuntimeAssets:
    config_path: Path
    checkpoint_path: Path
    vocab_path: Path


def _pick_first(paths: list[Path]) -> Path | None:
    return paths[0] if paths else None


def discover_assets(
    project_root: Path,
    configured_checkpoint: Path,
    configured_vocab: Path,
    configured_config: Path | None = None,
) -> RuntimeAssets:
    checkpoints_root = project_root / "checkpoints" / "slrt"
    default_config = project_root / "third_party_SLRT" / "Online" / "CSLR" / "configs" / "slide_phoenix-2014t.yaml"

    ckpt = configured_checkpoint if configured_checkpoint.exists() else None
    if ckpt is None and checkpoints_root.exists():
        ckpt = _pick_first(sorted(checkpoints_root.rglob("*.ckpt")))
    if ckpt is None and checkpoints_root.exists():
        ckpt = _pick_first(sorted(checkpoints_root.rglob("*.pth")))
    if ckpt is None and checkpoints_root.exists():
        ckpt = _pick_first(sorted(checkpoints_root.rglob("*.pt")))

    vocab = configured_vocab if configured_vocab.exists() else None
    if vocab is None and checkpoints_root.exists():
        vocab = _pick_first(sorted(checkpoints_root.rglob("*.vocab")))
    if vocab is None and checkpoints_root.exists():
        vocab = _pick_first(sorted(checkpoints_root.rglob("*.json")))

    config_candidates: list[Path] = []
    if configured_config is not None:
        config_candidates.append(configured_config)
    config_candidates.append(default_config)
    if checkpoints_root.exists():
        config_candidates.extend(sorted(checkpoints_root.rglob("*.yaml")))
        config_candidates.extend(sorted(checkpoints_root.rglob("*.yml")))
    config = _pick_first([p for p in config_candidates if p.exists()])

    if config is None:
        raise FileNotFoundError("No SLRT config file found. Provide SLR_SLRT_CONFIG_PATH.")
    if ckpt is None:
        raise FileNotFoundError(
            "No SLRT checkpoint found in checkpoints/slrt. Put a pretrained *.ckpt there or set SLRT_CHECKPOINT_PATH."
        )
    if vocab is None:
        raise FileNotFoundError(
            "No vocabulary file found in checkpoints/slrt. Put *.vocab or *.json there or set SLRT_VOCAB_PATH."
        )

    logger.info("SLRT assets config=%s checkpoint=%s vocab=%s", config, ckpt, vocab)
    return RuntimeAssets(config_path=config, checkpoint_path=ckpt, vocab_path=vocab)


def load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)
