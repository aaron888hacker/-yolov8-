

# 🧍‍♂️ YOLOv8-Pose Real-time Pose Detection System
**⭐ If this project is helpful to you, please consider giving it a Star!**

A graphical pose estimation tool based on YOLOv8-pose, supporting **image, camera, and video file** inputs. It detects human keypoints in real-time, draws skeletons, and identifies fine-grained poses such as **normal sitting, head bowing, slouching, and body tilting**.

![Python Version](https://img.shields.io/badge/python-3.8%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Ultralytics](https://img.shields.io/badge/Ultralytics-YOLOv8-orange)

---

## 📌 Features

- 🖼️ **Multiple Input Sources**: Images, cameras, local video files (.mp4/.avi/.mov)
- 🧠 **High-Accuracy Pose Estimation**: Based on YOLOv8-pose, supports 17 COCO keypoints
- 🎯 **Fine-grained Sitting Posture Analysis**:
  - Normal sitting posture
  - Head bowing (neck bending angle)
  - Slouching/Hunchback (torso forward tilt angle)
  - Left tilt / Right tilt (shoulder height difference)
- 🖥️ **User-friendly GUI**: Tkinter interface with real-time adjustable confidence, IoU, and pose judgment thresholds
- 🚀 **Real-time Performance**: Runs smoothly in camera mode (30fps+)
- 👥 **Multi-person Detection**: Supports simultaneous recognition and annotation of multiple humans in the frame

---

## 🖼️ Interface Preview

```
┌─────────────────────────────────────────────────┐
│  [Load Model]  Model loaded    Conf: ███░░      │
│  Tilt: ██░░░   Hunch: ██░░░    Bowed: ██░░░     │
│  [Open Image] [Open Video] [Start Camera] [Stop]│
├─────────────────────────────────────────────────┤
│                                                 │
│              【Real-time Video/Image Display Area】               │
│                                                 │
├─────────────────────────────────────────────────┤
│  2 person(s) detected                           │
│  Status: Camera started                         │
└─────────────────────────────────────────────────┘
```

> During actual runtime, keypoints, skeletons, and posture labels for each person will be displayed.

---

## 📦 Installation & Dependencies

### Environment Requirements
- Python 3.8 or higher
- Anaconda or a virtual environment is recommended

### Installation Steps

1. **Clone the Repository**
   ```bash
   git clone https://github.com/aaron888hacker/-yolov8-.git
   cd -yolov8-
   ```

2. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```
   `requirements.txt` contents:
   ```
   ultralytics>=8.0.0
   opencv-python>=4.5.0
   numpy>=1.21.0
   Pillow>=9.0.0
   ```

3. **Download Pre-trained Model**  
   Place `yolov8n-pose.pt` (or `yolov8s-pose.pt` / `yolov8m-pose.pt`) in the project root directory.  
   Download link: [Ultralytics Assets](https://github.com/ultralytics/assets/releases)

---

## 🚀 Quick Start

1. **Run the GUI**
   ```bash
   python src/main.py
   ```

   > Run from the repository root so the relative paths of `yolov8n-pose.pt` and `times.ttf` resolve.

2. **Load Model**: Click `Load Model` and wait for the "Model loaded" prompt.

3. **Select Input Source**:
   - **Image**: `Open Image` → Select an image containing a person
   - **Video File**: `Open Video` → Select a .mp4/.avi file
   - **Camera**: `Start Camera` → Open the default camera

4. **Adjust Thresholds** (takes effect in real-time):
   - `Conf`: Detection confidence (lowering it can detect smaller/blurred humans)
   - `Tilt thresh`: Sensitivity for body tilt judgment
   - `Hunch (°)`: Hunchback angle threshold (values above this are judged as hunchback)
   - `Bowed (°)`: Head bowing angle threshold (values above this are judged as head bowing)

5. **Stop**: Click `Stop` to close the current video/camera.

---

## ⚙️ Configuration & Customization

### Change Default Model
Modify `self.model_path` in `YOLOv8Pose.__init__`:
```python
self.model_path = 'yolov8m-pose.pt'   # Use medium model to improve accuracy
```

### Improve Small Object Detection
Increase the `imgsz` parameter in `YOLOv8Pose.process_frame`:
```python
results = self.model.predict(..., imgsz=1280)  # Default is 640
```
> Note: Higher resolution will increase GPU/CPU load.

### Add New Pose Categories
Modify the decision logic in `YOLOv8Pose.analyze_sitting_pose`, or extend the return values of `determine_pose`.

### Enable Chinese Labels
Change the `fine_pose` string to Chinese (e.g., `'低头'`), and ensure a Chinese-supporting font is used (e.g., `times.ttf`). In `process_frame`, you can replace `cv2.putText` with PIL drawing (refer to code comments).

---

## 📂 File Structure

```
.
├── src/main.py            # Main program (GUI + pose estimation core)
├── debug_pose.py          # Pose feature debugging script
├── yolov8n-pose.pt        # Pre-trained model
├── requirements.txt       # Dependency list (Python 3.8+)
├── times.ttf / STSONG.TTF # CJK fonts
├── res.json               # Sample detection output
├── docs/index.html        # GitHub Pages landing page
├── .github/workflows/     # Daily syntax and lint checks
├── README.md              # Chinese README
└── README.en-US.md        # English README (this file)
```

---

## 📊 Pose Labels Explanation

| Label (English)        | Meaning           | Judgment Criteria                     |
|-------------------|----------------|-----------------------------|
| standing          | Standing          | Knees straight, hips higher than knees       |
| sitting           | Sitting (Standard)    | Knees bent, hips higher than knees       |
| sitting_low       | Low sitting         | Knees bent and positioned low          |
| leaning           | Slight bend/Leaning      | Knee angle between 125°~160°       |
| normal sitting    | Normal sitting       | Sitting posture with all angles within normal range      |
| bowed head        | Head bowing          | Neck bending angle > threshold          |
| hunchback         | Hunchback/Slouching          | Torso forward tilt angle > threshold          |
| left tilt         | Body left tilt     | Right shoulder higher than left shoulder                |
| right tilt        | Body right tilt     | Left shoulder higher than right shoulder                |
| squat             | Squat          | Knee angle < 90°             |
| unknown           | Unable to determine       | Keypoint confidence insufficient            |

---

## 🧪 Test Environment

- OS: Windows 10/11, Ubuntu 20.04
- Python: 3.11.4
- GPU: NVIDIA RTX 3060 (Optional, runs on CPU as well)
- Camera: Standard USB camera / Laptop built-in camera

---

## 🤝 Contributing Guidelines

Issues and Pull Requests are welcome. If you wish to add new features (e.g., posture logging, alert notifications, etc.), please open an Issue for discussion first.

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).

---

## 🙏 Acknowledgments

- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) – Provides powerful pose estimation models
- [OpenCV](https://opencv.org/) – Image processing
- [Pillow](https://python-pillow.org/) – Image display

---
## Star History

<a href="https://www.star-history.com/?repos=aaron888hacker%2F-yolov8-&type=date&legend=top-left">
 <picture>
   <source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/chart?repos=aaron888hacker/-yolov8-&type=date&theme=dark&legend=top-left" />
   <source media="(prefers-color-scheme: light)" srcset="https://api.star-history.com/chart?repos=aaron888hacker/-yolov8-&type=date&legend=top-left" />
   <img alt="Star History Chart" src="https://api.star-history.com/chart?repos=aaron888hacker/-yolov8-&type=date&legend=top-left" />
 </picture>
</a>

## 📧 Contact

For questions or suggestions, please provide feedback via [GitHub Issues](https://github.com/aaron888hacker/-yolov8-/issues).

---

**⭐ If this project is helpful to you, please consider giving it a Star!**
