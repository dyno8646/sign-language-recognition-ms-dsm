# Repository Validation

## Scope

Validated these sources:

1. `https://github.com/FangyunWei/SLRT/tree/main/Online`
2. `https://github.com/FangyunWei/SLRT/tree/main/NLA-SLR`

## Findings

### `Online` (Primary for MVP)

- Contains explicit online/sliding inference implementations:
  - `Online/CSLR/prediction_slide.py`
  - `Online/CTC_fusion/prediction_online.py`
- Uses windowed temporal inference (`win_size`, `stride`) and CTC-style decoding.
- Supports continuous-sequence recognition logic, which best matches webcam streaming goals.
- Includes checkpoint-based testing flows and vocabulary-based gloss decoding.

### `NLA-SLR` (Secondary/Reference)

- Focused on isolated sign recognition and language-assisted training/eval:
  - `NLA-SLR/prediction.py`
  - `NLA-SLR/training.py`
- Designed around preprocessed dataset pipelines and distributed eval settings.
- Useful for:
  - checkpoint/config conventions
  - class semantics
  - post-classification ideas
- Not the fastest direct fit for near-real-time webcam sentence-like streaming.

## Dependencies & Runtime Observations

- Both repositories target older PyTorch-era stacks (e.g., `torch==1.9.0+cu102`) and legacy CUDA assumptions.
- Scripts are tightly coupled to dataset loaders and expected metadata structure.
- Direct "webcam in -> text out" is not provided as a turnkey endpoint.

## Compatibility Summary

- GPU: preferred for real-time throughput.
- CPU: possible for MVP with low FPS, reduced resolution, and larger inference interval.
- Online CSLR scripts assume model checkpoints and vocab/config files are prepared.

## Practical MVP Decision

- Use `Online` as primary inference blueprint and adapter source.
- Use `NLA-SLR` only for references and optional future improvement.
- Keep MediaPipe optional for ROI/keypoint preprocessing only; not as recognition model.

## Honest Limitations

- The upstream repos are research-oriented, not plug-and-play web inference services.
- Pretrained checkpoints must be manually acquired and mapped correctly to local config.
- Latency and accuracy depend heavily on checkpoint-dataset match and device capability.
