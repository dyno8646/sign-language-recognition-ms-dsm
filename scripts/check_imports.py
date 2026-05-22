from __future__ import annotations

import importlib


def mod_version(name: str) -> str:
    module = importlib.import_module(name)
    return getattr(module, "__version__", "unknown")


def main() -> None:
    modules = [
        "cv2",
        "torch",
        "torchvision",
        "fastapi",
        "httpx",
        "timm",
        "yaml",
        "numpy",
    ]
    print("== Import Check ==")
    for name in modules:
        v = mod_version(name)
        print(f"{name}: {v}")

    import torch

    print("cuda_available:", torch.cuda.is_available())
    print("torch_device:", "cuda" if torch.cuda.is_available() else "cpu")


if __name__ == "__main__":
    main()
