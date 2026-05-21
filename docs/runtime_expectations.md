# Runtime Commands and Expectations

## Commands

Install dependencies:

```bash
pip install -r requirements.txt
```

Run backend:

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Verify pipeline:

```bash
python scripts/verify_pipeline.py
```

## Checkpoint Locations

Runtime auto-discovers from:

- `checkpoints/slrt/**/*.ckpt|*.pth|*.pt`
- `checkpoints/slrt/**/*.vocab|*.json`
- config from `SLR_SLRT_CONFIG_PATH` or upstream `third_party_SLRT/Online/CSLR/configs/slide_phoenix-2014t.yaml`

## Expected Throughput

- Capture FPS: ~`3-5` (default `4`)
- Inference trigger: every `8` frames (default)
- Prediction cadence: roughly every `2-3` seconds

## Typical Latency Range

- GPU laptop: ~`150-600 ms` per inference window
- CPU fallback: ~`800-3000 ms` per inference window

## Estimated Memory

- VRAM (CUDA): ~`2-5 GB` depending on checkpoint/model variant and window size
- System RAM (CPU path): ~`3-8 GB` working set

These are practical MVP expectations; exact values depend on checkpoint stream configuration and hardware.
