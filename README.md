# MotionCoder 👐⚡

**Real-time Sign & Gesture → Intent → Action for CAD/DCC & VR Workflows**

> Turn human motion (hands, signs, gestures, short pantomime) into **precise, low-latency commands** for tools like **Blender** — with optional **text descriptions** of motion semantics.

[![License: Apache-2.0](https://img.shields.io/badge/License-Apache--2.0-blue.svg)](#-license) [![Status: MVP](https://img.shields.io/badge/status-MVP--planning-yellow)]() [![Python 3.11+](https://img.shields.io/badge/Python-3.11+-green)]()

---

## ✨ Key Features

* **Real-time intent inference** from **3D hand poses** (start with Leap Motion; multi-cam later).
* **Gesture → Command mapping** via a simple **Intent-DSL (JSON)**, e.g.:

  ```json
  { "intent": "mesh.cut", "params": { "plane_normal": [0,0,1], "offset_mm": 5 } }
  ```
* **Two learning modes**

  * **Supervised** (standard): train on labeled sequences.
  * **Self-Supervised Pretraining** + **Supervised Fine-Tuning**: learn useful representations first, then adapt with few labels.
* **Blender bridge** (MVP): invoke `bpy` operators, Geometry Nodes, gizmos; **low latency** event pipeline.
* **Pluggable backends**: start with **ST-GCN / PoseC3D (PyTorch)**; optional **new PyTorch re-implementation** for full control.
* **Accessible by design**: gestures/signs as a **first-class interface**; optional speech/text feedback.

---

## 🗺️ Architecture (MVP)

```
LeapC ──► Leap2Pose ──► MotionCoder (Encoder + Policy) ──► Coder2Blender ──► Blender API
                         ▲                     │
                         └────── Pose2Gizmos ──┘   (visual feedback / debug)
```

* **LeapC**: retrieves 3D hand joints (21/hand) at 60–120 Hz.
* **Leap2Pose**: normalization (root/scale), sequencing `C×T×J`.
* **MotionCoder**: ML encoder → **intent + slots** (+ hysteresis/policy).
* **Coder2Blender**: maps intents to `bpy` actions (event-driven, undo-safe).
* **Pose2Gizmos**: lightweight 3D overlays for feedback and tuning.

> **Future:** Replace Leap with **MVMono3D** (multi-view mono cameras, HW-sync, IR) exporting the same Pose API.

---

## 📦 Repo Layout (proposed)

```
motioncoder/
├─ apps/
│  ├─ blender_addon/           # Coder2Blender + Pose2Gizmos (UI overlay)
│  ├─ inference_service/       # FastAPI/WebSocket inference server
│  └─ leap_adapter/            # Leap2Pose (IPC to inference)
├─ motioncoder/
│  ├─ models/                  # ST-GCN / PoseC3D wrappers; optional new PyTorch impl
│  ├─ data/                    # datasets, transforms, loaders
│  ├─ policy/                  # hysteresis, debouncing, uncertainty handling
│  ├─ intent/                  # JSON schema, routing
│  └─ utils/                   # timing, logs, profiling
├─ configs/                    # training & runtime configs (yaml)
├─ scripts/                    # train, eval, export, benchmark
├─ docs/                       # guides, privacy, model card
└─ tests/
```

---

## 🚀 Quick Start (MVP with Leap + Blender)

### 1) Prerequisites

* **OS:** Ubuntu 22.04 / Windows 10+
* **Python:** 3.11+
* **Blender:** 4.x (Python API)
* **Leap Motion Controller 2** (Ultraleap) + **Leap SDK (C)**
* **GPU:** NVIDIA recommended (CUDA 12.x) for training/inference

### 2) Install

```bash
# clone
git clone https://github.com/kostukovic/motioncoder.git
cd motioncoder

# create env
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# install python deps
pip install -U pip wheel
pip install -r requirements.txt   # torch, torchvision, mmcv, mmpose/mmaction2 deps, fastapi, pydantic, onnxruntime, etc.
```

> If you use **MMAction2 / PySKL** recipes, follow their extra install notes (CUDA/torch matching).

### 3) Run the Inference Service

```bash
# start WebSocket/HTTP service (loads a default gesture model)
python apps/inference_service/main.py --config configs/runtime/leap_mvp.yaml
```

### 4) Start Leap Adapter

```bash
# reads LeapC, normalizes, sends sequences to inference
python apps/leap_adapter/main.py --ipc ws://127.0.0.1:8765
```

### 5) Install the Blender Add-on

* In Blender → **Edit → Preferences → Add-ons → Install…**
* Choose `apps/blender_addon/motioncoder_addon.zip` (or directory in dev mode).
* Configure **endpoint** (`ws://127.0.0.1:8765`) and enable **Gizmos**.

Now perform a few **iconic gestures** (e.g., “circle”, “cut”, “move 5 mm”) and watch **commands fire** in Blender ✨

---

## 🧪 Datasets & Formats

* **Input**: pose sequences as **`(C × T × J)`** or **`(T × J × C)`** (configurable), where

  * `C ∈ {2D, 3D}`, `T` = frames, `J` = joints (e.g., 21/hand).
* **Normalization**: root-relative, scale-normalized, temporal smoothing.
* **Labels**: `gesture_class`, optional **slots** (e.g., numeric parameters like distances, angles).
* **Storage**: `.npy/.pt` tensors + JSON sidecars for metadata.

> Use `scripts/convert_leap_logs.py` to build training/eval splits from recorded sessions (WIP).

---

## 🤖 Training Modes

### 1) Supervised (standard)

Train on labeled sequences (recommended to bootstrap the MVP).

```bash
python scripts/train.py \
  --config configs/train/stgcn_supervised.yaml \
  DATASET.PATH data/gestures_leap/ \
  TRAIN.EPOCHS 40
```

### 2) Self-Supervised Pretraining ➜ Supervised Fine-Tuning

Learn motion representations without labels, then adapt with few labels.

```bash
# self-supervised pretraining (e.g., masked-frame prediction / contrastive)
python scripts/pretrain_ssl.py \
  --config configs/train/ssl_pretrain.yaml \
  DATASET.PATH data/unlabeled_sessions/

# fine-tune
python scripts/finetune.py \
  --config configs/train/finetune_ssl.yaml \
  DATASET.PATH data/gestures_leap_labeled/ \
  TRAIN.EPOCHS 20
```

SSL backbones: **ST-GCN encoder**, **PoseC3D encoder**, or a **new PyTorch encoder** (optional).
Objectives: temporal contrast, masked joint/time, rotation prediction, etc.

---

## 🧭 Inference & Policy

**Why a policy?** Raw logits can “flutter”. We add:

* **Hysteresis & debouncing**
* **Confidence gates** & **hold-to-confirm** for critical ops
* **Cancel gesture** (safety)
* **Left-handed profile** & thresholds per user

```python
from motioncoder.policy import StablePolicy

policy = StablePolicy(min_hold_ms=120, enter=0.75, exit=0.55)
for frame in stream:
    probs = model(frame)                 # torch tensor [num_classes]
    event = policy.update(probs)         # returns None or Intent(...)
    if event:
        send_intent(event.to_json())
```

---

## 🔌 Intent-to-Action (Blender)

**Intent JSON → Action table** is configurable:

```yaml
# configs/intents/blender.yaml
routes:
  mesh.cut:
    operator: object.modifier_add
    params:
      type: "BOOLEAN"
    post:
      - operator: object.modifier_apply
        params: { modifier: "Boolean" }

  curve.draw_circle:
    operator: curve.primitive_bezier_circle_add
    params: { radius: "@slot.radius" }  # map slot value
```

> Keep actions **event-driven** (don’t mutate heavy meshes per frame). Use Gizmos & Nodes for smooth UX.

---

## 🧰 Roadmap

* ✅ **MVP**: Leap → Pose → MotionCoder → Blender bridge
* ⏭️ **Personalization**: quick calibration; few-shot/LoRA fine-tuning
* ⏭️ **Unreal bridge** (tech preview → pilot)
* ⏭️ **MVMono3D** (multi-cam, HW-sync, IR) with same Pose API
* ⏭️ **ONNX/TensorRT** export for low-latency deployment
* ⏭️ **Model Card & Privacy docs**; opt-in telemetry (lat/FPS only)

---

## 🧑‍🏫 Usage Guide (Leitfaden)

**Phase 1 — Try**

1. Plug in **Leap** → install SDK.
2. Run **Inference Service** + **Leap Adapter**.
3. Install **Blender Add-on**; enable Gizmos.
4. Perform **6–8 core gestures** (circle, line, cut, rotate, nudge, confirm/cancel).

**Phase 2 — Tune**
5. Adjust **policy thresholds** until UX feels stable.
6. Map **intents to actions** you use daily (edit the YAML routes).
7. Record a short session; **review events overlay**.

**Phase 3 — Train**
8. Label a few minutes of your own gestures → **supervised training**.
9. Optional: collect unlabeled streams → **SSL pretraining** → fine-tune.
10. Export to **ONNX** (and later TensorRT) for faster inference.

**Phase 4 — Integrate**
11. Add **custom gizmos** / Geometry Nodes for parametric edits.
12. Share your config via PR; help grow the **Intent catalog**.

---

## 🛡️ Privacy, Ethics & Safety (short)

* **On-prem by default**; no cloud required.
* No biometric ID; focus on **gesture content**, not persons.
* Telemetry is **opt-in** and aggregates **latency/FPS only**.
* Undo-safe actions, **hold-to-confirm** for destructive ops.
* See `docs/privacy.md`, `docs/model_card.md`, `docs/toms.md`.

---

## ❓ FAQ

**Q: Do I need a VR headset?**
A: No. Large monitor + Leap is the MVP. Headsets can be explored later.

**Q: Can I start without cameras?**
A: Yes — **Leap** provides 3D hand skeletons immediately.

**Q: Why not 2D keypoints from webcams first?**
A: Works for demos, but jitter/occlusion are limiting for **precise CAD**. Multi-view mono (future) solves that.

**Q: PyTorch re-implementation vs. existing frameworks?**
A: We’ll **start with proven stacks (ST-GCN/PoseC3D)** for speed. A **clean PyTorch re-implementation** is planned if/when we need full control and custom SSL.

---

## 🤝 Contributing

* **Good first issues:** add a new **intent mapping**, improve a **gizmo**, write a **dataset converter**.
* Please run tests & linters; follow the **Contributor Guide** (coming with the first public alpha).
* Code of Conduct applies to all repos under **Apache-2.0**.

---

## 📚 Related

* **MVMono3D** (future multi-cam 3D): *link placeholder*
* **Blender**: [https://www.blender.org/](https://www.blender.org/)
* **MMAction2 / PySKL / PoseC3D**: upstream training/inference recipes

---

## 🗺️ Status & Milestones (short)

* **M1:** repo/CI, Leap2Pose, baseline inference, Blender add-on skeleton.
* **M2:** 6–8 gestures, stable policy, 10 reproducible Blender tasks.
* **M3:** MVP v0.9 release, quick calibration, pilot A.
* **M4–M6:** ergonomics, public alpha, pilots & impact measurement.

Detailed plan lives in `docs/roadmap.md`.

---

## 🧾 License

**Apache-2.0** — free & permissive for FOSS and commercial use.
See `LICENSE`.

---

## 💬 Acknowledgments

* Open-source communities in **PyTorch**, **OpenMMLab (MMAction2/MMPose)**, and **Blender**.
* Accessibility & signing communities inspiring **gesture-first** interfaces.

---

**Ready to iterate?** Create your first **intent mapping**, tune thresholds, and record a mini-demo. PRs welcome! 🚀
