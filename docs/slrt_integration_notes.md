# SLRT Integration Notes (Inference Only)

## Why adapter mode is needed

`FangyunWei/SLRT` online scripts are research pipelines tied to dataset metadata and specific config hierarchies.  
For a webcam MVP, we use a service adapter architecture and keep inference runtime pluggable.

## Current Adapter Behavior

- `SLR_ENGINE=mock`:
  - always available
  - stable demo flow for UI/backend validation
- `SLR_ENGINE=slrt_online`:
  - validates checkpoint and vocab loading
  - prepares frame tensor pipeline
  - designed hook point for true model forward + CTC decode

## Required assets to enable true SLRT decoding

- compatible checkpoint (`best.ckpt`)
- matching `vocab.json`
- matching model/config assumptions from SLRT `Online` pipeline
- optional keypoint stream handling if checkpoint expects two-stream input

## CPU fallback guidance

- lower capture resolution (`320x240` or less)
- use lower FPS (`2-4`)
- increase inference interval (`inference_every_n_frames`)
- keep sliding window size moderate (`8-16`)
