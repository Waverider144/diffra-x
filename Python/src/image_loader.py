import cv2
import numpy as np

def load_target_image(config: dict, shape=(256, 256)) -> np.ndarray:
    """根据配置加载或生成目标图像 (归一化到 0-1)"""
    mode = config.get("image_source", "text")
    
    if mode == "file":
        img = cv2.imread(config["image_path"], cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(f"无法读取图像: {config['image_path']}")
        img = cv2.resize(img, shape)
    elif mode == "text":
        # 动态生成文字图像
        img = np.zeros(shape, dtype=np.uint8)
        text = config.get("target_text", "ODE")
        font = cv2.FONT_HERSHEY_SIMPLEX
        
        # 统一使用 1.5 的字号进行计算和绘制
        font_scale = 1.5
        thickness = 3
        
        # 1. 计算该字号下的实际像素大小
        text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
        
        # 2. 算得精确的居中坐标
        text_x = (shape[1] - text_size[0]) // 2
        text_y = (shape[0] + text_size[1]) // 2
        
        # 3. 使用完全相同的参数绘制
        cv2.putText(img, text, (text_x, text_y), font, font_scale, 255, thickness, cv2.LINE_AA)
    else:
        raise ValueError(f"未知的图像来源模式: {mode}")
        
    return img.astype(np.float32) / 255.0