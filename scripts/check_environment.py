from __future__ import annotations

import platform

import torch


def main() -> None:
    print("Platform:", platform.platform())
    print("Python:", platform.python_version())
    print("Torch:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("CUDA device:", torch.cuda.get_device_name(0))


if __name__ == "__main__":
    main()
