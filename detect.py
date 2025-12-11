import cv2
import csv
import os
import numpy as np
from datetime import datetime
from ultralytics import YOLO

# --- CONFIGURATION ---
model_path = 'best.pt'

# 1. STRICTER THRESHOLD (Fixes "Ghost" detections)
conf_threshold = 0.55 

# 2. PIXEL-TO-MM CALIBRATION 
# Updated to 4.0 based on your "1.45mm" error. This should be closer to reality.
pixels_per_mm = 4.0  

# 3. SIZE FILTERS
min_grain_area = 100 
max_grain_area = 5000 

# CSV SETUP
csv_file = "Rice_Quality_Report.csv"
if not os.path.exists(csv_file):
    with open(csv_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Timestamp", "Total", "Whole", "Broken", "Foreign", "Avg Length (mm)", "Quality %", "Image File"])

print("Loading AI model... please wait...")
try:
    model = YOLO(model_path)
    print("✅ Model loaded successfully!")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    exit()

# CAMERA AUTO-SEARCH
found_camera = False
cap = None
for index in [1, 2, 3]:
    print(f"Testing Camera Index {index}...")
    cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
    if cap.isOpened():
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        ret, frame = cap.read()
        if ret:
            print(f"✅ FOUND CAMERA at Index {index}!")
            found_camera = True
            break
        else:
            cap.release()

if not found_camera:
    print("❌ Error: Could not find DroidCam.")
    exit()

print("✅ SYSTEM READY! (Erosion + Strict Mode Active)")

while True:
    success, frame = cap.read()
    if not success: break

    # 1. AI Inference
    results = model(frame, conf=conf_threshold, verbose=False)

    # 2. DATA PROCESSING
    whole_rice = 0
    broken_rice = 0
    foreign_obj = 0
    grain_lengths_px = []
    
    final_frame = frame.copy()

    for result in results:
        for box in result.boxes:
            # Get Box Coordinates
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            w = x2 - x1
            h = y2 - y1
            area = w * h

            # --- FILTER 1: SIZE CHECK ---
            if area < min_grain_area or area > max_grain_area:
                continue

            # Get Label
            class_id = int(box.cls[0])
            label_name = model.names[class_id]

            # Logic: If it touches, counting might fail, but we save stats.
            if "sound" in label_name:
                whole_rice += 1
            elif "broken" in label_name:
                broken_rice += 1
            elif "foreign" in label_name:
                foreign_obj += 1

            # --- SMART GEOMETRY ANALYSIS (The Erosion Fix) ---
            if "foreign" not in label_name:
                # 1. Cut out the grain image
                roi = frame[y1:y2, x1:x2]
                if roi.size == 0: continue

                # 2. Convert to Binary (Black & White)
                gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                _, mask = cv2.threshold(gray_roi, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
                
                # 3. EROSION (Digital Sandpaper)
                # This shrinks the grain slightly to break connections with neighbors
                kernel = np.ones((3,3), np.uint8)
                eroded_mask = cv2.erode(mask, kernel, iterations=1)

                # 4. Find the shape inside the eroded mask
                contours, _ = cv2.findContours(eroded_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                if contours:
                    # Find the biggest blob (the actual rice grain)
                    c = max(contours, key=cv2.contourArea) 
                    
                    # Measure it
                    rect = cv2.minAreaRect(c)
                    (center, (w_rect, h_rect), angle) = rect
                    length_px = max(w_rect, h_rect)
                    grain_lengths_px.append(length_px)

            # Draw Box
            color = (0, 255, 0) # Green default
            if "broken" in label_name or "foreign" in label_name:
                color = (0, 0, 255) # Red

            cv2.rectangle(final_frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(final_frame, label_name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # 3. CALCULATE AVERAGE SIZE
    avg_length_mm = 0.0
    if len(grain_lengths_px) > 0:
        avg_px = sum(grain_lengths_px) / len(grain_lengths_px)
        avg_length_mm = round(avg_px / pixels_per_mm, 2)

    # 4. DASHBOARD
    total_rice = whole_rice + broken_rice
    quality_score = 0
    if total_rice > 0:
        quality_score = int((whole_rice / total_rice) * 100)

    # Overlay Box
    overlay = final_frame.copy()
    cv2.rectangle(overlay, (5, 5), (280, 200), (0, 0, 0), -1) 
    
    # Red Alert
    if foreign_obj > 0:
        cv2.rectangle(overlay, (5, 5), (280, 200), (0, 0, 255), -1)
        cv2.putText(final_frame, "CONTAMINATION!", (300, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 3)

    final_frame = cv2.addWeighted(overlay, 0.6, final_frame, 0.4, 0)
    
    # Stats Text
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(final_frame, f"TOTAL:  {total_rice}", (15, 35), font, 0.8, (255, 255, 255), 2)
    cv2.putText(final_frame, f"Whole:  {whole_rice}", (15, 70), font, 0.6, (0, 255, 0), 2)
    cv2.putText(final_frame, f"Broken: {broken_rice}", (15, 100), font, 0.6, (0, 0, 255), 2)
    cv2.putText(final_frame, f"Avg Size:{avg_length_mm} mm", (15, 140), font, 0.7, (0, 255, 255), 2)
    cv2.putText(final_frame, f"Quality: {quality_score}%", (15, 180), font, 0.6, (200, 200, 200), 1)

    cv2.imshow("Rice Quality Inspector (Pro)", final_frame)

    # 5. CONTROLS
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'): break
    elif key == ord('s'):
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        image_filename = f"rice_scan_{timestamp}.jpg"
        cv2.imwrite(image_filename, final_frame)
        with open(csv_file, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([timestamp, total_rice, whole_rice, broken_rice, foreign_obj, avg_length_mm, f"{quality_score}%", image_filename])
        print(f"✅ Saved Report: {image_filename}")
        white_flash = cv2.addWeighted(final_frame, 0.5, 255 * (final_frame > 0).astype('uint8'), 0.5, 0)
        cv2.imshow("Rice Quality Inspector (Pro)", white_flash)
        cv2.waitKey(100)

cap.release()
cv2.destroyAllWindows()