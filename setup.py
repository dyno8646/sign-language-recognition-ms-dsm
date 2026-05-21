"""
Setup script for Sign Language Recognition (MS DSM capstone).

Install editable (recommended for development):
    pip install -e .

Note: The local package folder ``mediapipe/`` shadows Google's ``mediapipe`` pip
package. Use Python 3.11 and ``pip install mediapipe==0.10.14`` for Holistic tracking.
Prefer running scripts from the project root (see README).
"""

from pathlib import Path

from setuptools import find_packages, setup

ROOT = Path(__file__).resolve().parent


def read_readme() -> str:
    """Load long description from README.md."""
    readme = ROOT / "README.md"
    if readme.is_file():
        return readme.read_text(encoding="utf-8")
    return "Real-time sign language recognition — MS DSM capstone project."


def read_requirements() -> list[str]:
    """Parse requirements.txt, skipping comments and empty lines."""
    req_file = ROOT / "requirements.txt"
    if not req_file.is_file():
        return []

    requirements: list[str] = []
    for line in req_file.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        requirements.append(line)
    return requirements


setup(
    name="sign-language-recognition-ms-dsm",
    version="0.1.0",
    author="MS DSM Project",
    description=(
        "Real-time sign language recognition: webcam, MediaPipe Holistic, "
        "BiLSTM, gloss decoding, LLM refinement, and TTS."
    ),
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/dyno8646/sign-language-recognition-ms-dsm",
    license="MIT",
    python_requires=">=3.10,<3.13",
    packages=find_packages(
        where=".",
        exclude=[
            "venv",
            "venv.*",
            "venv311",
            "venv311.*",
            "scripts",
            "notebooks",
            "datasets",
            "checkpoints",
            "configs",
            "docker",
            "data",
            "src",
            "frontend",
        ],
    ),
    include_package_data=True,
    install_requires=read_requirements(),
    extras_require={
        "dev": [
            "pytest>=8.0.0",
            "ruff>=0.4.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "slr-capture=webcam.webcam_capture:main",
            "slr-train=training.train_bilstm:main",
            "slr-infer=inference.realtime_inference:main",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    keywords="sign-language recognition mediapipe bilstm asl computer-vision",
)
