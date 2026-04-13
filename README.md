# Intelligent Vision System for Visually Impaired Navigation

![Status](https://img.shields.io/badge/Status-Code_Release_Pending-FFA500.svg)
![YOLOv8](https://img.shields.io/badge/YOLO-v8_Nano-00FFFF.svg)
![EdgeAI](https://img.shields.io/badge/Hardware-Raspberry_Pi_4-C51A4A.svg)

## Overview
This repository showcases a wearable, low-latency Edge AI system designed to assist visually impaired individuals in complex urban environments where traditional mobility aids fail to detect head-level obstacles or distant dynamic hazards such as cyclists, silent electric vehicles, etc. 

This project bridges geospatial engineering and scalable data science by fusing **Semantic Object Detection** (What is it?) with **Monocular Metric Depth Estimation** (How far is it?), translating visual data into a priority-based spatial audio warning system.

### Live Demonstration

https://github.com/user-attachments/assets/36c317c8-3e8f-4b3a-bf5d-b3c35d295e43

## System Architecture & Methodology
The system employs a dual-stream AI pipeline engineered for resource-constrained hardware:
1. **Vision Stream:** A custom fine-tuned `YOLOv8-Nano` model trained on the **BDD100K Dataset**, explicitly targeting 7 safety-critical street classes: `Bike, Bus, Car, Motor, Person, Rider, Truck`.
2. **Depth Stream:** `DepthAnything V2 (Metric Outdoor)`, providing zero-shot metric distance estimation without requiring active LiDAR sensors.
3. **Sensor Fusion:** Bounding boxes from YOLO act as dynamic region-of-interest masks over the depth map. The system calculates the median pixel depth to determine the object's absolute distance, rejecting background noise.

## Edge Hardware Optimization
To achieve real-time inference (~10-15 FPS) on a **Raspberry Pi 4**, the models were rigorously optimized:
* **YOLOv8n:** Quantized from FP32 PyTorch to INT8 precision using the **NCNN** framework, reducing memory footprint by over 50%.
* **DepthAnything:** Exported to an optimized **ONNX** runtime engine.

## Performance & Results
The system utilizes a 3-tier proximity audio alert system (Notice: >10m, Warning: <5m, Critical: <2m) to prevent cognitive overload.

Validated on the BDD100K hold-out set, the model achieved an overall **mAP@50 of 0.429**. 
* **Car Detection Precision:** 73.4%
* **Truck Detection Precision:** 50.7%

*Note: The system prioritizes Recall over Precision to ensure maximum user safety, as a false positive is preferable to a missed obstacle.*

---
**Code Availability:** *The source code, quantized models, and deployment scripts are currently being formatted and will be pushed to this repository shortly.*
