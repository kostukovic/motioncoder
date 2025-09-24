# MotionCoder

**LLM-powered motion understanding for real-time gesture → command pipelines**
*From 3D keypoints & ROI point clouds to structured commands for CAD/DCC/VR.*

[![License: Apache-2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](#license) [![Status: MVP in progress](https://img.shields.io/badge/status-MVP_in_progress-yellow)](#roadmap)

---

## What is MotionCoder?

**MotionCoder** is a modular engine that turns **spatiotemporal motion** (3D keypoints + sparse ROI point clouds) into **structured outputs** (JSON/DSL, code, text). It’s designed for **real-time** human gestures and other motions, and integrates with **CAD/DCC/VR** via lightweight plugins.

* **Input:** Multi-view 3D keypoints (+ optional ROI point clouds)
* **Core:** Motion encoder + LLM head (semantics, validation, uncertainty)
* **Output:** JSON/DSL commands, text descriptions, or API events for downstream tools
* **Target latency:** ≤ **80–100 ms** end-to-end on a modern NVIDIA GPU

---

## Key features

* **LLM semantics for motion:** map gestures/signs to **precise commands** (e.g., *“extrude 5 mm”*, *“split edge”*).
* **3D-first:** robust **multi-view triangulation** / depth → 3D keypoints and compact **ROI** context.
* **Realtime by design:** CUDA/TensorRT deployment, streaming gRPC/IPC, zero-copy data path.
* **Extensible:** clean **DSL/JSON** schema, plugin hooks for CAD/DCC (Blender, FreeCAD, etc.).
* **Open**: Apache-2.0 license, clear model/data cards and governance.

---

## Why not just 2D?

2D keypoints lifted to 3D work for many cases but degrade with **occlusions/extreme poses**. MotionCoder embraces **calibrated multi-view** and/or **depth** to keep **metric accuracy** and **temporal stability**, which matters for **industrial CAD** and **precise animation**.

---

## Architecture (high level)

```
Cameras / Sensors ─┬─> 2D keypoints
                   └─> Depth / ROI patches  ──┐
Multi-View Calib & Triangulation  ────────────┼─> 3D keypoints (+ ROI) ──┐
Temporal filters (One-Euro / Kalman)          ┘                          │
                                                                         v
                          Motion Encoder (4D: points × time)  ──>  LLM Head
                                                          ┌───────────┬─────────────┐
                                                          │           │             │
                                                    JSON/DSL Cmds   Text        Embeddings
                                                          │
                                                      gRPC/IPC
                                                          │
                                                  CAD/DCC/VR Plugins
```

---

## Example output (JSON/DSL)

```json
{
  "timestamp": "2025-09-22T12:03:41.512Z",
  "actor": "hand.right",
  "gesture": "draw_circle",
  "params": { "radius_mm": 24.8, "plane": "screen" },
  "confidence": 0.93,
  "roi_contacts": ["tool.tip ~ surface.A"],
  "intent": "sketch_circle",
  "actions": [
    { "cmd": "sketch.begin" },
    { "cmd": "sketch.circle", "args": { "r_mm": 25 } },
    { "cmd": "sketch.commit" }
  ]
}
```

---

## Quick start

> **Prereqs:** Linux (Ubuntu LTS recommended), Python 3.11+, CUDA 12+, recent NVIDIA GPU.
> Optional: **MC2PCM** (multi-camera → point cloud) for live capture, or use recorded data.

```bash
# 1) Clone
git clone https://github.com/<you>/motioncoder.git
cd motioncoder

# 2) Create env
python -m venv .venv && source .venv/bin/activate
pip install -U pip

# 3) Install (training/inference)
pip install -e ".[dev]"

# 4) Run a demo (offline sample → JSON events)
python demos/run_offline.py \
  --input samples/sequence_01.npz \
  --config configs/motioncoder_base.yaml \
  --out events.jsonl
```

**TensorRT deployment (optional):**

```bash
# Export to ONNX and build TRT engine
python tools/export_onnx.py --ckpt checkpoints/motioncoder.ckpt
python tools/build_trt.py --onnx out/motioncoder.onnx --fp16
```

---

## Repo layout (suggested)

```
motioncoder/
  configs/                # model, IO, plugin routing
  demos/                  # quick demos (offline/streaming)
  motioncoder/
    io/                   # readers (NPZ, ROSbag, FlatBuffers), gRPC/IPC
    models/               # encoder backbones, LLM heads, adapters/LoRA
    runtime/              # TensorRT serving, schedulers, filters
    semantics/            # DSL/JSON schema, validators, intent mapping
    utils/                # timing, metrics, viz helpers
  plugins/
    blender/              # example CAD/DCC connector
    freecad/
  docs/
    model_card.md
    data_flow.md
    privacy.md
  tests/
```

---

## Data & inputs

* **3D Keypoints:** from multi-view triangulation (OpenCV/ChArUco, EasyMocap/Anipose-like) or depth sensors.
* **ROI point clouds:** sparse local patches around hands/tools/contacts for geometric context.
* **Synthetic first, real next:** start with **synthetic motion** (e.g., *AnimalMotionCoder*) to stabilize encoder; fine-tune on human gestures/signs.

---

## Roadmap

* **MVP (now):** 4D motion encoder + LLM head, JSON/DSL events, ≤100 ms E2E on single GPU.
* **M1–M3:** multi-task heads (class/attributes/status), FP16/INT8, temporal filters.
* **M4:** TensorRT serving, confidence & OOD handling.
* **M5:** CAD/DCC plugins (Blender/FreeCAD), command mapping library.
* **M6:** Dataset/tooling for sign/gesture semantics; public model & docs.

See issues for granular milestones.

---

## Plugins & integration

* **gRPC/IPC** streaming events
* **SDKs:** TypeScript / Python (planned)
* **Targets:** Blender, FreeCAD first; adapters for Maya/Unreal/Fusion 360 can follow.

---

## Performance targets

* **Latency:** ≤ 80–100 ms @ ≥ 30 FPS (E2E capture→events)
* **Accuracy:** class macro-F1 ≥ 0.90 on internal test sets (initially synthetic)
* **Robustness:** occlusion & viewpoint stress-tests; calibrated multi-view preferred

---

## Contributing

Contributions welcome!
Please read **`CONTRIBUTING.md`** and the **Code of Conduct**.

* Discuss big changes in an issue first.
* Add tests/docs for new features.
* Follow semantic versioning (API/DSL).

---

## License

**Apache-2.0**. See [`LICENSE`](./LICENSE).

---

## Related projects

* **MC2PCM** — Multi-Camera-to-Point-Cloud Module (capture, calibration, triangulation).
* **AnimalMotionCoder** — synthetic motion set & benchmarks for fast encoder bootstrapping.

---

## Acknowledgements

Built with **PyTorch**, **TensorRT**, **OpenCV**, **Ceres**, and community 3D/vision stacks.

---

If you want, I can also add:

* a minimal **demo script** (`demos/run_offline.py`) that emits JSON events,
* a **DSL schema** (`docs/command_schema.json`) and
* a **Blender plugin stub** to complete the starter kit.
