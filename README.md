# 🌾 Rice Quality Inspector AI (YOLOv8 + OpenCV)

### **Automated Grain Analysis, Adulteration Detection & Quality Auditing System**

![Project Status](https://img.shields.io/badge/Status-Prototype%20Complete-success)
![Python](https://img.shields.io/badge/Python-3.12-blue)
![Computer Vision](https://img.shields.io/badge/Computer%20Vision-YOLOv8%20%7C%20OpenCV-orange)

## 🚀 Project Overview
This project is an **AI-powered Quality Control (QC) system** designed to automate the inspection of rice grains. Traditional manual inspection is slow, subjective, and prone to error. This system uses **Deep Learning (YOLOv8)** for object detection and **Geometric Computer Vision (OpenCV)** to analyze grain morphology in real-time.

It is capable of detecting **broken grains**, **discoloration**, and **foreign contaminants** (stones/shells), while simultaneously calculating the **Average Grain Length (AGL)** to identify variety (e.g., Basmati vs. Non-Basmati).

## 🎯 Key Features
* **Real-Time Detection:** Identifies Whole vs. Broken grains with >95% accuracy using a custom-trained YOLOv8 model.
* **Geometric Analysis:** Calculates precise grain length (mm) using `cv2.minAreaRect` and **Morphological Erosion** to separate touching grains.
* **Strict Mode:** Filters out "ghost" detections (shadows/dust) using area thresholds and confidence logic.
* **Contamination Alert:** Instant visual "Red Alert" upon detecting foreign objects (stones, plastic).
* **Automated Auditing:** Generates a `.csv` report and saves image snapshots for every batch inspected (`Rice_Quality_Report.csv`).

## 🛠️ Tech Stack
* **Core:** Python 3.12
* **AI Model:** YOLOv8 (Ultralytics) - Trained on labeled grain images.
* **Vision Library:** OpenCV (Morphological operations, Contours, Pixel-to-Metric conversion).
* **Hardware Integration:** DroidCam (USB Low-Latency Mode) for high-res image acquisition.

## 📊 How It Works
1.  **Acquisition:** Video feed is captured via USB interface.
2.  **Preprocessing:** Frames are resized and filtered for noise using morphological erosion.
3.  **Inference:** YOLOv8 detects objects (`sound`, `broken`, `foreign`).
4.  **Post-Processing:**
    * **Strict Filters:** Removes ghost detections based on confidence (>0.55) and area thresholds.
    * **Geometry:** Applies `cv2.erode` to separate touching grains and measures pixel length.
    * **Calibration:** Converts pixels to millimeters using a reference factor.
5.  **Reporting:** Live dashboard overlay + CSV logging.

## ⚙️ Setup & Installation
1.  **Clone the Repository:**
    ```bash
    git clone [https://github.com/Roboteer7/Rice-Quality-AI-Inspector.git](https://github.com/Roboteer7/Rice-Quality-AI-Inspector.git)
    cd Rice-Quality-AI-Inspector
    ```
2.  **Install Dependencies:**
    ```bash
    pip install ultralytics opencv-python numpy
    ```
3.  **Hardware Setup:**
    * Connect phone via USB using DroidCam Client.
    * Ensure rice is placed on a matte black background for optimal contrast.

## 🚀 Usage
Run the main detection script:
```bash
python detect.py



Controls:
s Key: Save Report (Captures photo + Logs stats to CSV).

q Key: Quit the application.

🔧 Calibration
The system uses a pixel-to-metric conversion factor (pixels_per_mm). To calibrate for a new camera height:

Place a ruler under the camera.

Adjust the pixels_per_mm variable in detect.py until the "Avg Size" matches the ruler.

