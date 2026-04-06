import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import threading
import time
from ultralytics import YOLO
from PIL import ImageDraw, ImageFont  # Image可能已有，补全ImageDraw和ImageFont

# -------------------- 姿态估计核心类（整合之前优化版）--------------------
class YOLOv8Pose:
    def __init__(self, model_path='yolov8n-pose.pt', device='cpu', conf=0.25, iou=0.7):
        self.device = device
        self.conf = conf
        self.iou = iou
        self.model = YOLO(model_path)
        print(f"[INFO] 模型 '{model_path}' 已加载。")

    def detect(self, img_source):
        results = self.model.predict(
            source=img_source,
            conf=self.conf,
            iou=self.iou,
            device=self.device,
            verbose=False
        )
        return results

    @staticmethod
    def determine_pose(keypoints):
        if len(keypoints) < 17:
            return '未知'
        key_indices = [11, 12, 13, 14, 15, 16]
        if any(keypoints[i][2] < 0.3 for i in key_indices):
            return '未知'
        def calculate_angle(a, b, c):
            ba = np.array(a[:2]) - np.array(b[:2])
            bc = np.array(c[:2]) - np.array(b[:2])
            dot = np.dot(ba, bc)
            norm = np.linalg.norm(ba) * np.linalg.norm(bc)
            cos_theta = np.clip(dot / norm, -1.0, 1.0)
            return np.degrees(np.arccos(cos_theta))
        left_knee_angle = calculate_angle(keypoints[11], keypoints[13], keypoints[15])
        right_knee_angle = calculate_angle(keypoints[12], keypoints[14], keypoints[16])
        avg_knee_angle = (left_knee_angle + right_knee_angle) / 2
        avg_hip_y = (keypoints[11][1] + keypoints[12][1]) / 2
        avg_knee_y = (keypoints[13][1] + keypoints[14][1]) / 2
        knee_below_hip = avg_knee_y > avg_hip_y
        if avg_knee_angle < 125:
            if knee_below_hip:
                if avg_knee_angle < 90:
                    return '深蹲'
                else:
                    return '坐姿 (低角度)'
            else:
                return '坐姿'
        elif 125 <= avg_knee_angle < 160:
            return '微屈/倚靠'
        else:
            if knee_below_hip:
                return '站立'
            else:
                return '坐姿 (腿伸直)'

    @staticmethod
    def analyze_sitting_pose(keypoints):
        if len(keypoints) < 17:
            return '坐姿(未知)'
        nose = keypoints[0][:2]
        left_ear = keypoints[3][:2]
        right_ear = keypoints[4][:2]
        left_shoulder = keypoints[5][:2]
        right_shoulder = keypoints[6][:2]
        left_hip = keypoints[11][:2]
        right_hip = keypoints[12][:2]
        shoulder_mid = (left_shoulder + right_shoulder) / 2
        hip_mid = (left_hip + right_hip) / 2
        ear_mid = (left_ear + right_ear) / 2

        def angle_between(v1, v2):
            v1_u = v1 / (np.linalg.norm(v1) + 1e-6)
            v2_u = v2 / (np.linalg.norm(v2) + 1e-6)
            dot = np.clip(np.dot(v1_u, v2_u), -1.0, 1.0)
            return np.degrees(np.arccos(dot))

        vec_neck = ear_mid - shoulder_mid
        vertical = np.array([0, 1])
        neck_angle = angle_between(vec_neck, vertical)

        vec_torso = shoulder_mid - hip_mid
        torso_angle = angle_between(vec_torso, vertical)

        shoulder_diff = left_shoulder[1] - right_shoulder[1]
        shoulder_width = np.linalg.norm(left_shoulder - right_shoulder)
        shoulder_tilt_ratio = shoulder_diff / (shoulder_width + 1e-6)

        # 使用GUI传入的阈值（稍后通过实例变量获取）
        # 这里先保留默认值，实际会在process_frame中根据GUI阈值调整
        return neck_angle, torso_angle, shoulder_tilt_ratio  # 改为返回数值，决策在process_frame中做

    def process_frame(self, frame):
        """处理单帧图像，返回带标注的图像和检测人数"""
        orig_frame = frame.copy()
        results = self.model.predict(source=frame, conf=self.conf, iou=self.iou,
                                     device=self.device, verbose=False)
        if len(results) == 0 or results[0].keypoints is None or len(results[0].boxes) == 0:
            return orig_frame, "未检测到人"

        result = results[0]
        num_people = len(result.boxes)
        skeleton = [
            (0,1),(0,2),(1,3),(2,4),(3,5),(4,6),
            (5,6),(5,7),(7,9),(6,8),(8,10),
            (5,11),(6,12),(11,12),
            (11,13),(12,14),(13,15),(14,16)
        ]

        # 获取GUI中设置的阈值
        tilt_thresh = self.tilt_thresh if hasattr(self, 'tilt_thresh') else 0.1
        torso_thresh = self.torso_thresh if hasattr(self, 'torso_thresh') else 20
        neck_thresh = self.neck_thresh if hasattr(self, 'neck_thresh') else 40

        # 用于存储需要绘制的中文文本及其位置
        text_items = []

        for i in range(num_people):
            keypoints_array = result.keypoints.data[i].cpu().numpy()
            base_pose = self.determine_pose(keypoints_array)

            # 如果是坐姿，进行细粒度分析
            if '坐姿' in base_pose:
                neck_angle, torso_angle, shoulder_tilt_ratio = self.analyze_sitting_pose(keypoints_array)
                # 应用阈值进行决策
                if abs(shoulder_tilt_ratio) > tilt_thresh:
                    if shoulder_tilt_ratio > 0:
                        fine_pose = '右侧斜'
                    else:
                        fine_pose = '左侧斜'
                elif torso_angle > torso_thresh:
                    fine_pose = '驼背'
                elif neck_angle > neck_thresh:
                    fine_pose = '低头'
                else:
                    fine_pose = '正常坐姿'
            else:
                fine_pose = base_pose

            # 绘制边界框（OpenCV）
            box = result.boxes.xyxy[i].cpu().numpy().astype(int)
            cv2.rectangle(orig_frame, (box[0], box[1]), (box[2], box[3]), (255,0,0), 2)

            # 记录要绘制的文本位置和内容（稍后用PIL批量绘制）
            text_items.append(((box[0], box[1]-10), fine_pose))

            # 绘制关键点
            for kpt in keypoints_array:
                x, y, conf = int(kpt[0]), int(kpt[1]), kpt[2]
                if conf > 0.5:
                    cv2.circle(orig_frame, (x, y), 3, (0,255,0), -1)

            # 绘制骨架
            for start, end in skeleton:
                if start < len(keypoints_array) and end < len(keypoints_array):
                    if keypoints_array[start][2] > 0.5 and keypoints_array[end][2] > 0.5:
                        x1, y1 = int(keypoints_array[start][0]), int(keypoints_array[start][1])
                        x2, y2 = int(keypoints_array[end][0]), int(keypoints_array[end][1])
                        cv2.line(orig_frame, (x1, y1), (x2, y2), (0,255,0), 2)

        # 批量绘制中文文本（解决问号问题）
        if text_items:
            # 将OpenCV图像转换为PIL图像
            img_rgb = cv2.cvtColor(orig_frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(img_rgb)
            draw = ImageDraw.Draw(pil_img)
            # 设置中文字体（请确保字体文件存在，可替换为实际路径）
            font = ImageFont.truetype('times.ttf', 20, encoding='utf-8')  # 或 'simhei.ttf' 等
            for pos, text in text_items:
                draw.text(pos, text, font=font, fill=(0, 255, 255))  # 黄色
            # 转回OpenCV图像
            orig_frame = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

        return orig_frame, f"检测到 {num_people} 人"


# -------------------- Tkinter GUI --------------------
class PoseGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("姿态检测系统 - 坐姿分析 (支持多人)")
        self.root.geometry("900x700")
        self.root.resizable(True, True)

        # 初始化检测器
        self.pose_estimator = None
        self.model_path = 'yolov8n-pose.pt'
        self.device = 'cpu'
        self.conf_thresh = tk.DoubleVar(value=0.25)
        self.iou_thresh = tk.DoubleVar(value=0.7)
        self.tilt_thresh = tk.DoubleVar(value=0.1)
        self.torso_thresh = tk.DoubleVar(value=20)
        self.neck_thresh = tk.DoubleVar(value=40)

        # 视频/摄像头相关
        self.cap = None
        self.video_running = False
        self.video_thread = None

        # 创建界面
        self.create_widgets()

    def create_widgets(self):
        # 顶部控制区
        control_frame = ttk.Frame(self.root, padding=10)
        control_frame.pack(fill=tk.X)

        # 模型加载
        ttk.Button(control_frame, text="加载模型", command=self.load_model).grid(row=0, column=0, padx=5)
        self.model_status = ttk.Label(control_frame, text="模型未加载", foreground="red")
        self.model_status.grid(row=0, column=1, padx=5)

        # 阈值调节
        ttk.Label(control_frame, text="置信度:").grid(row=0, column=2, padx=5)
        ttk.Scale(control_frame, from_=0.1, to=0.9, variable=self.conf_thresh,
                  orient=tk.HORIZONTAL, length=100).grid(row=0, column=3)
        ttk.Label(control_frame, text="倾斜阈值:").grid(row=0, column=4, padx=5)
        ttk.Scale(control_frame, from_=0.05, to=0.3, variable=self.tilt_thresh,
                  orient=tk.HORIZONTAL, length=100).grid(row=0, column=5)

        # 第二行控制
        ttk.Label(control_frame, text="驼背阈值(°):").grid(row=1, column=0, padx=5)
        ttk.Scale(control_frame, from_=10, to=40, variable=self.torso_thresh,
                  orient=tk.HORIZONTAL, length=100).grid(row=1, column=1)
        ttk.Label(control_frame, text="低头阈值(°):").grid(row=1, column=2, padx=5)
        ttk.Scale(control_frame, from_=20, to=60, variable=self.neck_thresh,
                  orient=tk.HORIZONTAL, length=100).grid(row=1, column=3)

        # 操作按钮
        ttk.Button(control_frame, text="打开图片", command=self.open_image).grid(row=1, column=4, padx=5)
        ttk.Button(control_frame, text="打开摄像头", command=self.open_camera).grid(row=1, column=5, padx=5)
        ttk.Button(control_frame, text="停止", command=self.stop_video).grid(row=1, column=6, padx=5)

        # 图像显示区域
        self.image_label = ttk.Label(self.root)
        self.image_label.pack(expand=True, fill=tk.BOTH, padx=10, pady=10)

        # 姿态结果标签
        self.result_label = ttk.Label(self.root, text="姿态: 无", font=("微软雅黑", 16))
        self.result_label.pack(pady=5)

        # 状态栏
        self.status_bar = ttk.Label(self.root, text="就绪", relief=tk.SUNKEN, anchor=tk.W)
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)

    def load_model(self):
        try:
            self.pose_estimator = YOLOv8Pose(model_path=self.model_path, device=self.device,
                                              conf=self.conf_thresh.get(), iou=self.iou_thresh.get())
            self.model_status.config(text="模型已加载", foreground="green")
            self.status_bar.config(text="模型加载成功")
        except Exception as e:
            messagebox.showerror("错误", f"模型加载失败: {str(e)}")

    def open_image(self):
        if self.pose_estimator is None:
            messagebox.showwarning("警告", "请先加载模型")
            return
        file_path = filedialog.askopenfilename(filetypes=[("图像文件", "*.jpg *.jpeg *.png")])
        if not file_path:
            return
        self.status_bar.config(text=f"正在处理: {file_path}")
        # 读取图像
        frame = cv2.imread(file_path)
        if frame is None:
            messagebox.showerror("错误", "无法读取图像")
            return
        # 更新阈值到检测器
        self.pose_estimator.conf = self.conf_thresh.get()
        self.pose_estimator.iou = self.iou_thresh.get()
        # 将GUI阈值传递给检测器实例（用于process_frame）
        self.pose_estimator.tilt_thresh = self.tilt_thresh.get()
        self.pose_estimator.torso_thresh = self.torso_thresh.get()
        self.pose_estimator.neck_thresh = self.neck_thresh.get()
        # 处理
        processed_frame, info = self.pose_estimator.process_frame(frame)
        # 显示
        self.display_image(processed_frame)
        self.result_label.config(text=info)
        self.status_bar.config(text="处理完成")

    def open_camera(self):
        if self.pose_estimator is None:
            messagebox.showwarning("警告", "请先加载模型")
            return
        if self.video_running:
            return
        self.cap = cv2.VideoCapture(0)  # 0为默认摄像头
        if not self.cap.isOpened():
            messagebox.showerror("错误", "无法打开摄像头")
            return
        self.video_running = True
        self.video_thread = threading.Thread(target=self.video_loop, daemon=True)
        self.video_thread.start()
        self.status_bar.config(text="摄像头已开启")

    def stop_video(self):
        self.video_running = False
        if self.cap:
            self.cap.release()
        self.status_bar.config(text="已停止")

    def video_loop(self):
        while self.video_running:
            ret, frame = self.cap.read()
            if not ret:
                break
            # 更新阈值
            self.pose_estimator.conf = self.conf_thresh.get()
            self.pose_estimator.iou = self.iou_thresh.get()
            self.pose_estimator.tilt_thresh = self.tilt_thresh.get()
            self.pose_estimator.torso_thresh = self.torso_thresh.get()
            self.pose_estimator.neck_thresh = self.neck_thresh.get()
            # 处理帧
            processed_frame, info = self.pose_estimator.process_frame(frame)
            # 显示（在主线程中更新UI）
            self.root.after(0, self.display_image, processed_frame)
            self.root.after(0, self.result_label.config, {"text": info})
            # 控制帧率
            time.sleep(0.03)  # ~30fps
        if self.cap:
            self.cap.release()

    def display_image(self, cv_image):
        """将OpenCV图像转换为Tkinter可显示的格式"""
        cv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
        # 缩放以适应窗口
        h, w = cv_image.shape[:2]
        max_size = 700
        if h > max_size or w > max_size:
            scale = max_size / max(h, w)
            new_w, new_h = int(w * scale), int(h * scale)
            cv_image = cv2.resize(cv_image, (new_w, new_h))
        img = Image.fromarray(cv_image)
        imgtk = ImageTk.PhotoImage(image=img)
        self.image_label.config(image=imgtk)
        self.image_label.image = imgtk  # 保持引用

# -------------------- 启动 --------------------
if __name__ == "__main__":
    root = tk.Tk()
    app = PoseGUI(root)
    root.mainloop()