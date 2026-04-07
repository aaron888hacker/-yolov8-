# 🧍‍♂️ YOLOv8-Pose 实时姿态检测系统
**⭐ 如果这个项目对你有帮助，欢迎给个 Star！**

基于 YOLOv8-pose 的图形化姿态估计工具，支持**图片、摄像头、视频文件**输入，实时检测人体关键点、绘制骨架，并识别**正常坐姿、低头、驼背、身体侧倾**等精细姿态。

![Python Version](https://img.shields.io/badge/python-3.8%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Ultralytics](https://img.shields.io/badge/Ultralytics-YOLOv8-orange)

---

## 📌 功能特点

- 🖼️ **多种输入源**：图片、摄像头、本地视频文件（.mp4/.avi/.mov）
- 🧠 **高精度姿态估计**：基于 YOLOv8-pose，支持 17 个 COCO 关键点
- 🎯 **精细坐姿分析**：
  - 正常坐姿
  - 低头（颈部弯曲角度）
  - 驼背（躯干前倾角度）
  - 左侧斜 / 右侧斜（肩膀高度差）
- 🖥️ **友好 GUI**：Tkinter 界面，可实时调节置信度、IOU、姿态判断阈值
- 🚀 **实时性能**：摄像头模式下可流畅运行（30fps+）
- 👥 **多人检测**：支持画面中多个人体同时识别与标注

---

## 🖼️ 界面预览

```
┌─────────────────────────────────────────────────┐
│  [Load Model]  Model loaded    Conf: ███░░      │
│  Tilt: ██░░░   Hunch: ██░░░    Bowed: ██░░░    │
│  [Open Image] [Open Video] [Start Camera] [Stop]│
├─────────────────────────────────────────────────┤
│                                                 │
│              【实时视频/图片显示区域】           │
│                                                 │
├─────────────────────────────────────────────────┤
│  2 person(s) detected                           │
│  Status: Camera started                         │
└─────────────────────────────────────────────────┘
```

> 实际运行时会显示每个人体的关键点、骨架和姿态标签。

---

## 📦 安装与依赖

### 环境要求
- Python 3.8 或更高版本
- 建议使用 Anaconda 或虚拟环境

### 安装步骤

1. **克隆仓库**
   ```bash
   git clone https://github.com/aaron888hacker/-yolov8-.git
   cd yolov8-pose-gui
   ```

2. **安装依赖包**
   ```bash
   pip install -r requirements.txt
   ```
   `requirements.txt` 内容：
   ```
   ultralytics>=8.0.0
   opencv-python>=4.5.0
   numpy>=1.21.0
   Pillow>=9.0.0
   ```

3. **下载预训练模型**  
   将 `yolov8n-pose.pt`（或 `yolov8s-pose.pt` / `yolov8m-pose.pt`）放入项目根目录。  
   下载地址：[Ultralytics Assets](https://github.com/ultralytics/assets/releases)

---

## 🚀 快速开始

1. **运行 GUI**
   ```bash
   python pose_gui.py
   ```

2. **加载模型**：点击 `Load Model`，等待提示“Model loaded”。

3. **选择输入源**：
   - **图片**：`Open Image` → 选择一张包含人物的图片
   - **视频文件**：`Open Video` → 选择 .mp4/.avi 文件
   - **摄像头**：`Start Camera` → 打开默认摄像头

4. **调节阈值**（实时生效）：
   - `Conf`：检测置信度（降低可识别更小/模糊的人体）
   - `Tilt thresh`：身体侧倾判断灵敏度
   - `Hunch (°)`：驼背角度阈值（大于此值判为驼背）
   - `Bowed (°)`：低头角度阈值（大于此值判为低头）

5. **停止**：点击 `Stop` 关闭当前视频/摄像头。

---

## ⚙️ 配置与自定义

### 修改默认模型
在 `PoseGUI.__init__` 中更改 `self.model_path`：
```python
self.model_path = 'yolov8m-pose.pt'   # 改用 medium 模型提升精度
```

### 提高小目标检测能力
在 `PoseEstimator.process_frame` 中增加 `imgsz` 参数：
```python
results = self.model.predict(..., imgsz=1280)  # 默认640
```
> 注意：更高的分辨率会增加 GPU/CPU 负载。

### 添加新的姿态类别
修改 `PoseEstimator.analyze_sitting_pose` 的决策逻辑，或扩展 `determine_pose` 的返回值。

### 启用中文标签
将 `fine_pose` 字符串改为中文（如 `'低头'`），并确保使用支持中文的字体（如 `simhei.ttf`）。在 `process_frame` 中可替换 `cv2.putText` 为 PIL 绘图（参考代码注释）。

---

## 📂 文件结构

```
.
├── pose_gui.py            # 主程序（GUI + 姿态估计核心）
├── yolov8n-pose.pt        # 预训练模型
├── requirements.txt       # 依赖列表
├── README.md              # 本文件

```

---

## 📊 姿态标签说明

| 标签（英文）        | 含义           | 判断依据                     |
|-------------------|----------------|-----------------------------|
| standing          | 站立           | 膝盖伸直，髋部高于膝盖       |
| sitting           | 坐姿（标准）   | 膝盖弯曲，髋部高于膝盖       |
| sitting_low       | 低坐姿         | 膝盖弯曲且位置较低           |
| leaning           | 微屈/倚靠      | 膝盖角度在125~160之间        |
| normal sitting    | 正常坐姿       | 坐姿且各角度在正常范围内     |
| bowed head        | 低头           | 颈部弯曲角度 > 阈值          |
| hunchback         | 驼背           | 躯干前倾角度 > 阈值          |
| left tilt         | 身体左侧斜     | 右肩高于左肩                 |
| right tilt        | 身体右侧斜     | 左肩高于右肩                 |
| squat             | 深蹲           | 膝盖角度 < 90°               |
| unknown           | 无法判断       | 关键点置信度不足             |

---

## 🧪 测试环境

- OS: Windows 10/11, Ubuntu 20.04
- Python: 3.11.4
- GPU: NVIDIA RTX 3060 (可选，CPU 也可运行)
- 摄像头: 普通 USB 摄像头 / 笔记本内置摄像头
- 
---

## 🤝 贡献指南

欢迎提交 Issue 和 Pull Request。如果您想添加新功能（如姿态记录、报警提示等），请先开 Issue 讨论。

---

## 📄 许可证

本项目采用 [MIT 许可证](LICENSE)。

---

## 🙏 致谢

- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) – 提供强大的姿态估计模型
- [OpenCV](https://opencv.org/) – 图像处理
- [Pillow](https://python-pillow.org/) – 图像显示

---

## 📧 联系方式

如有问题或建议，请通过 [GitHub Issues](https://github.com/aaron888hacker/yolov8-pose-gui/issues) 反馈。

---

**⭐ 如果这个项目对你有帮助，欢迎给个 Star！**
