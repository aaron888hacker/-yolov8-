import cv2
import numpy as np
from ultralytics import YOLO

class PoseDebugger:
    def __init__(self, model_path='yolov8n-pose.pt'):
        self.model = YOLO(model_path)
    
    def extract_features(self, keypoints):
        """从17个关键点中提取核心特征，用于区分坐姿与站姿"""
        if keypoints.shape != (17, 3):
            return {}
        
        # 中点计算
        hip_center = (keypoints[11][:2] + keypoints[12][:2]) / 2
        knee_center = (keypoints[13][:2] + keypoints[14][:2]) / 2
        ankle_center = (keypoints[15][:2] + keypoints[16][:2]) / 2
        
        # 1. 躯干倾斜度 (脊柱角度): 髋中心 -> 肩中心 (关键点5,6的中点)
        shoulder_center = (keypoints[5][:2] + keypoints[6][:2]) / 2
        spine_vector = shoulder_center - hip_center
        spine_angle = np.degrees(np.arctan2(abs(spine_vector[0]), spine_vector[1])) if spine_vector[1] != 0 else 90
        
        # 2. 膝盖弯曲角度
        def angle_between(v1, v2):
            dot = np.dot(v1, v2)
            norm = np.linalg.norm(v1) * np.linalg.norm(v2)
            return np.degrees(np.arccos(np.clip(dot/norm, -1, 1)))
        
        knee_angle_left = angle_between(keypoints[11][:2]-keypoints[13][:2], 
                                        keypoints[15][:2]-keypoints[13][:2])
        knee_angle_right = angle_between(keypoints[12][:2]-keypoints[14][:2], 
                                         keypoints[16][:2]-keypoints[14][:2])
        avg_knee_angle = (knee_angle_left + knee_angle_right) / 2
        
        # 3. 关键高度比例 (判断是否“坐下”)
        # 坐下时，膝盖y坐标通常接近甚至高于髋部y坐标
        hip_y = hip_center[1]
        knee_y = knee_center[1]
        ankle_y = ankle_center[1]
        
        # 计算比例：膝盖高度 / 髋部高度 （值越大，膝盖相对位置越低）
        knee_to_hip_ratio = knee_y / hip_y if hip_y > 0 else 1
        ankle_to_knee_ratio = ankle_y / knee_y if knee_y > 0 else 1
        
        return {
            'spine_angle': spine_angle,          # 脊柱角度：越小越直立
            'avg_knee_angle': avg_knee_angle,    # 平均膝盖角度：站立时接近180°
            'knee_to_hip_ratio': knee_to_hip_ratio,  # >1：膝盖低于髋部；<1：膝盖高于髋部（可能为坐姿）
            'ankle_to_knee_ratio': ankle_to_knee_ratio, # >1：脚踝低于膝盖
            'hip_y': hip_y,
            'knee_y': knee_y,
            'ankle_y': ankle_y
        }
    
    def debug_image(self, img_path):
        """调试单张图片"""
        print(f"\n=== 调试图片: {img_path} ===")
        results = self.model.predict(img_path, verbose=False)
        
        for i, r in enumerate(results):
            if r.keypoints is None:
                continue
                
            for person_idx in range(len(r.boxes)):
                kpts = r.keypoints.data[person_idx].cpu().numpy()
                features = self.extract_features(kpts)
                
                print(f"\n人物 {person_idx+1}:")
                print(f"  置信度: {r.boxes.conf[person_idx]:.3f}")
                print(f"  特征值:")
                for key, val in features.items():
                    if 'angle' in key:
                        print(f"    {key}: {val:.1f}°")
                    elif 'ratio' in key:
                        print(f"    {key}: {val:.2f}")
                    elif '_y' in key:
                        print(f"    {key}: {val:.0f} (像素)")

# 使用示例
if __name__ == "__main__":
    debugger = PoseDebugger()
    # 请用您的图片路径替换
    test_images = ['stand.png', 'sit.png']  # 请准备明确的“站”和“坐”的测试图
    for img in test_images:
        debugger.debug_image(img)