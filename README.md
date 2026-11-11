# Intelligent Vision System for Visually Impaired Navigation

![YOLOv8](https://img.shields.io/badge/YOLO-v8_Nano-00FFFF.svg)
![EdgeAI](https://img.shields.io/badge/Hardware-Raspberry_Pi_4-C51A4A.svg)

Code and models accompanying the paper *"Enhancing Visually Impaired Navigation Using AI-driven Sensors And Edge Chip"* (A. M. Adetoro, B. P. Lijoodi, Technical University of Munich) as part of Big Geospatial Data Course, TU Munich.

## Overview
This project explores a wearable, low-latency edge-AI system prototype to help visually impaired people in urban environments, where a white cane misses head-level obstacles and fast or silent hazards such as cyclists and electric vehicles. It fuses **semantic object detection** (what is it?) with **monocular metric depth estimation** (how far is it?) and turns the result into distance-tiered audio warnings.

### Live demonstration
https://github.com/user-attachments/assets/36c317c8-3e8f-4b3a-bf5d-b3c35d295e43

## System architecture
1. **Vision stream:** `YOLOv8n` fine-tuned on a 7-class BDD100K subset: bike, bus, car, motor, person, rider, truck.
2. **Depth stream:** Depth Anything V2 Metric Outdoor (Small), which outputs depth in metres from a single RGB image, with no LiDAR or camera calibration required.
3. **Sensor fusion:** each YOLO box selects a region of the depth map; the **median** depth in that region is the object's distance (robust to background pixels inside the box). The nearest object drives the alert.
4. **Audio alerts:** the edge script (`vision_inference_script.py`) beeps slowly below 5 m (warning) and quickly below 2 m (critical). The desktop video demo adds a third, <10 m "notice" tier.

## Edge deployment
- Detector exported to **NCNN** (FP16) for CPU inference; depth model exported to **ONNX** and run with ONNX Runtime. Target hardware: Raspberry Pi 4 (4 GB).

## Results (validation split, 1,960 images; default Ultralytics settings)
Overall: **mAP@50 = 0.430**, mAP@50-95 = 0.246, precision 0.553, recall 0.398.

| Class | Precision | Recall | mAP@50 |
|---|---|---|---|
| car | 0.721 | 0.676 | 0.731 |
| truck | 0.525 | 0.502 | 0.506 |
| person | 0.609 | 0.452 | 0.494 |
| bus | 0.534 | 0.411 | 0.463 |
| bike | 0.512 | 0.303 | 0.332 |
| motor | 0.516 | 0.233 | 0.252 |
| rider | 0.451 | 0.210 | 0.231 |

Design intent is to favour recall over precision (a false alarm is preferable to a missed obstacle), but recall is currently low for small or rare classes (rider, motor, bike). Improving it is the main goal of future work. The validation split was also used to select the best checkpoint during training, so it is not a fully held-out set. On the untouched 980-image **test split** the same model scores mAP@50 = **0.436** (mAP@50-95 0.249, precision 0.560, recall 0.425; car 0.746, truck 0.541, person 0.477, bus 0.473, bike 0.291, rider 0.330, motor 0.198), evaluated with default Ultralytics settings. Figures: `docs/figures/` (validation split).

## Repository layout
| Path | Purpose |
|---|---|
| `vision_inference_script.py` | Headless edge pipeline (Raspberry Pi / Linux): NCNN YOLO + ONNX depth + `aplay` beeps |
| `bdd100k_7model_ncnn_model/` | NCNN export of the detector (used by the edge script) |
| `bdd100k_7model.pt` | Fine-tuned YOLOv8n weights (PyTorch) |
| `depth_model.onnx` | **Not in the repo** (95 MB). Download from the Releases page (`<https://github.com/Aderonke-Adetoro/visual-impairment-urban-navigation/releases/download/v1.0/depth_model.onnx>`) or regenerate it with the ONNX cell of `bdd100_7class.ipynb`; place it in the repository root |
| `bdd100_7class.ipynb` | Desktop notebook: video demo, **validation** (`model.val`, PR curve / confusion matrix), **Depth Anything V2 → ONNX export**, and **beep generation** |
| `training/BDD100k_Training.ipynb` | Colab training notebook (see notes below) |
| `docs/figures/` | PR curve and normalised confusion matrix from the validation split (default Ultralytics settings, mAP@50 = 0.430) |
| `beep_fast.wav`, `beep_slow.wav` | Alert sounds |

Run scripts from the repository root.

## Usage
Edge pipeline (Linux, needs `aplay` and a webcam). First obtain `depth_model.onnx` (see the table above):
```bash
pip install -r requirements.txt
python vision_inference_script.py
```
Desktop demos: `pip install -r requirements-demo.txt`, then open the notebooks. `bdd100_7class.ipynb` (video-demo cell) and add your own video to replace `long_street_clip.mp4`.

## Training details (from the checkpoint's embedded arguments)
YOLOv8n initialised from COCO `yolov8n.pt`; 40 epochs, early-stopping patience 10, batch 16, 640x640, seed 0, Ultralytics 8.3.248, Tesla T4. The optimiser was `optimizer=auto` (Ultralytics selected AdamW). Validation mAP@50: 0.430 (default settings, re-run; the checkpoint stores 0.428).

## Dataset
Training used the Roboflow Universe dataset [`eastsky/bdd100k-e3s18`](https://universe.roboflow.com/eastsky/bdd100k-e3s18) (v1, CC BY 4.0), a 7-class subset of BDD100K (bike, bus, car, motor, person, rider, truck) that is already split into train / valid / test (6,857 / 1,960 / 980 images). The dataset is not redistributed here; please download it from Roboflow in YOLOv8 format.

## Reproducing the pipeline
1. **Train:** `training/BDD100k_Training.ipynb` (Colab, T4) on the dataset above.
2. **Validate / plots:** `bdd100_7class.ipynb`, validation cell (`conf=0.25, iou=0.6`). The data path inside it is a local Windows path; edit it to your dataset location.
3. **Detector export (NCNN):** `YOLO('bdd100k_7model.pt').export(format='ncnn', half=True)` (Ultralytics 8.3.232, NCNN 1.0.20260114). This is FP16 - see the "Notes" below.
4. **Depth export:** `bdd100_7class.ipynb`, ONNX cell (`opset_version=14`, 1x3x518x518 input, `pixel_values` -> `predicted_depth`).
5. **Beeps:** `bdd100_7class.ipynb`, sound-generation cell (1500 Hz / 100 ms and 600 Hz / 300 ms sine waves).

## Notes and known discrepancies
- **Reported metrics.** Re-validated on the 1,960-image validation split with default Ultralytics settings (CPU, Ultralytics 8.4.208): mAP@50 = **0.430**, mAP@50-95 = 0.246, P = 0.553, R = 0.398 (car: P 0.721, R 0.676, mAP@50 0.731). The checkpoint's stored value is 0.428 (training-time validation). The `bdd100_7class.ipynb` validation cell (`conf=0.25, iou=0.6`) printed mAP@50 = 0.517, but this could not be reproduced: the same settings now give P 0.596, R 0.391 (matching the notebook) and mAP@50 = 0.343.
- **Danger tiers.** The paper describes <2 m / <5 m / <10 m. `vision_inference_script.py` implements <2 m and <5 m; the video-demo notebook implements all three.
- **Domain shift.** Trained only on outdoor street data; indoor webcam use produced unreliable class and distance estimates.
- Distances are approximate monocular estimates, not measurements, and this is a research prototype, not a certified safety device.

## Licence and attribution
This repository is released under the **GNU AGPL-3.0** (see `LICENSE`). The code loads models through Ultralytics YOLO (AGPL-3.0) and `bdd100k_7model.pt` / the NCNN export are derived from Ultralytics weights, so AGPL-3.0 is the licence that is compatible.

Third-party components:
- **Dataset:** Roboflow Universe `eastsky/bdd100k-e3s18`, CC BY 4.0, derived from BDD100K (see the BDD100K terms). Not redistributed here.
- **Depth model:** `depth-anything/Depth-Anything-V2-Metric-Outdoor-Small-hf` (Depth Anything V2). Check the model card for its current licence before redistributing `depth_model.onnx`.
- **Ultralytics YOLOv8:** AGPL-3.0.
