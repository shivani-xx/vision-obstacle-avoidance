# Vision-Based Obstacle Avoidance Using Lightweight CNNs on Edge Devices

## Research Internship — Purple AI Labs Ltd

**Intern:** Shivani  
**Mentor:** Mohan Lekshmanan

---

## Overview

This project investigates the trade-off between model size and navigation accuracy for vision-based obstacle avoidance in simulated environments.

The research compares lightweight CNN architectures for autonomous navigation under edge-device constraints, including:

- TinyCNN
- SqueezeNet
- ShuffleNet
- MobileNetV3
- EfficientNet

Evaluation includes:

- Accuracy benchmarking
- CPU latency benchmarking
- ONNX deployment export
- Real-time PyBullet navigation
- Multi-environment robustness testing

---

## Quick Start

```bash
pip install -r requirements.txt

python scripts/collect_data.py --difficulty medium --frames 5000

python scripts/run_training.py --model mobilenetv3 --size 224 --epochs 15

python scripts/run_benchmark.py
```

---

## Repository Structure

```text
vision-obstacle-avoidance/
├── configs/
├── src/
├── scripts/
├── notebooks/
├── results/
└── data/
```

---

## Results

See `results/` for:
- benchmark outputs
- JSON logs
- navigation metrics
- latency measurements

See `notebooks/analysis.ipynb` for charts and analysis.

---

## Models Evaluated

| Model | Parameters | Purpose |
|---|---|---|
| TinyCNN | ~24K | Minimal edge baseline |
| SqueezeNet | ~724K | Lightweight CNN |
| ShuffleNetV2 | ~1.2M | Mobile-efficient CNN |
| MobileNetV3-Small | ~1.5M | Edge-optimized CNN |
| EfficientNet-B0 | ~4.0M | Accuracy-focused CNN |

---

## Deployment

Models were exported to:
- PyTorch (.pth)
- ONNX (.onnx)

for edge-device deployment testing.

---

## Research Goals

- Investigate accuracy-latency tradeoffs
- Study robustness across environments
- Benchmark lightweight CNNs
- Explore deployment-oriented AI design