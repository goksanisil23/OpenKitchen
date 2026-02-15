# Conditional Flow Matching for BEV -> Action

This repository contains a simple, readable implementation of conditional flow matching:
- Condition: BEV image (`.png`)
- Target: single-step action (`throttle`, `steering`) from paired `.txt`

The model predicts normalized actions:
- `throttle_norm` in approximately `[-1, 1]` (from physical `0..100`)
- `steering_norm` in approximately `[-1, 1]` (from physical `-10..10` deg)

## Data format

Put your files in one folder as pairs with the same stem:
- `birdseye_SAOPAULO_993.png`
- `birdseye_SAOPAULO_993.txt`

Action txt file should contain one line with two numbers:
- `throttle steering`
- Example: `37.5 -2.1`

## Train

```bash
python train_flow_matching.py \
  --data-dir /path/to/your/data \
  --output-dir checkpoints \
  --epochs 30 \
  --batch-size 64
```

Outputs:
- `checkpoints/last.pt`
- `checkpoints/best.pt`
- `checkpoints/policy_scripted.pt` (after running exporter below)

Training always preloads all samples to GPU VRAM once at startup, then reuses those tensors across epochs.
CUDA is required for training.

## Inference

```bash
python inference_example.py \
  --checkpoint checkpoints/best.pt \
  --image /path/to/your/data/birdseye_SAOPAULO_993.png \
  --num-steps 32
```

This prints:
- normalized action (`throttle_norm`, `steering_norm`)
- denormalized physical action (`throttle`, `steering`)

## Export Scripted Model (for C++ later)

```bash
python export_scripted_model.py \
  --checkpoint checkpoints/best.pt \
  --output-scripted checkpoints/policy_scripted.pt
```

The exporter:
- loads trained weights
- creates a TorchScript model
- verifies scripted output matches original model output on random test batches
- saves `policy_scripted.pt`

## Main files

- `flow_matching_model.py`
  - `BEVEncoder`: image encoder
  - `ActionFlowTrunk`: flow network on action space
  - `ConditionalFlowMatchingPolicy`: combined model
  - `BEVActionDataset`: data loader for png/txt pairs
- `train_flow_matching.py`: training loop
- `inference_example.py`: standalone inference example
- `export_scripted_model.py`: TorchScript export + output equivalence check
