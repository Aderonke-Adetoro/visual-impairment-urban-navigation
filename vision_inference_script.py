import cv2
import numpy as np
import onnxruntime as ort
from ultralytics import YOLO
import time
import os

# CONFIGURATION
# Defining the 7 classes trained
CLASS_NAMES = {0: 'bike', 1: 'bus', 2: 'car', 3: 'motor', 4: 'person', 5: 'rider', 6: 'truck'}

# Audio Cooldown (Prevents system crash from too many aplay calls)
last_beep_time = 0
beep_cooldown = 0.5  # Seconds between beeps

# LOAD MODELS
print("Loading NCNN YOLO...")
model = YOLO('bdd100k_7model_ncnn_model') 

print("Loading ONNX Depth...")
depth_session = ort.InferenceSession("depth_model.onnx")

# CAMERA SETUP
cap = cv2.VideoCapture(0) # 0 for Webcam
cap.set(3, 640)
cap.set(4, 480)
width = int(cap.get(3))
height = int(cap.get(4))

print("Starting Vision System...")

while True:
    ret, frame = cap.read()
    if not ret: break

    # DEPTH (ONNX Way)
    # Resize to model input as requieed(518x518 is standard for V2 small)
    img_depth = cv2.resize(frame, (518, 518))
    img_depth = cv2.cvtColor(img_depth, cv2.COLOR_BGR2RGB)
    img_depth = img_depth.astype(np.float32) / 255.0
    img_depth = (img_depth - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225] # Normalize
    img_depth = np.transpose(img_depth, (2, 0, 1)) # HWC -> CHW
    img_depth = np.expand_dims(img_depth, axis=0) # Add batch dim

    # Run Inference
    depth_pred = depth_session.run(None, {"pixel_values": img_depth})[0]
    
    # Resize depth back to frame size so it matches YOLO coordinates
    depth_map = cv2.resize(depth_pred[0], (width, height)) # FIXED: Use width/height variables

    # YOLO (NCNN Way)
    results = model(frame, verbose=False)[0]

    # LOGIC
    min_dist = 999.0
    
    for box in results.boxes:
        cls_id = int(box.cls[0])
        if cls_id not in CLASS_NAMES: continue #
        
        x1, y1, x2, y2 = map(int, box.xyxy.cpu().numpy()[0])
        # Clamp to image boundaries
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(width, x2), min(height, y2)
        
        # Slicing the correct variable now
        obj_depth = depth_map[y1:y2, x1:x2] 
        
        if obj_depth.size > 0:
            dist = np.median(obj_depth)
            min_dist = min(min_dist, dist)

    #AUDIO LOGIC (With Safety Cooldown)
    current_time = time.time()
    
    if min_dist < 2.0:
        print(f" EMERGENCY: {min_dist:.1f}m")
        # Only beep if cooldown passed
        if (current_time - last_beep_time) > 0.2: 
            os.system("aplay beep_fast.wav &") # '&' runs it in background
            last_beep_time = current_time
            
    elif min_dist < 5.0:
        print(f" WARNING: {min_dist:.1f}m")
        if (current_time - last_beep_time) > 0.8:
            os.system("aplay beep_slow.wav &")
            last_beep_time = current_time

    # No cv2.imshow needed for headless chip