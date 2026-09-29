import cv2
import numpy as np
import torch

class TargetAnnealer:
    def __init__(self, config: dict, total_iters: int):
        self.start_sigma = config.get("anneal_start_sigma", 10.0)
        self.end_sigma = config.get("anneal_end_sigma", 0.1)
        self.total_iters = total_iters

    def get_target_at_step(self, base_img: np.ndarray, step: int) -> torch.Tensor:
        """根据当前迭代步数计算模拟退火后的目标图像"""
        # 指数衰减或线性衰减
        progress = step / max(1, self.total_iters - 1)
        sigma = self.start_sigma * (self.end_sigma / self.start_sigma) ** progress
        
        if sigma > 0.3: # 避免核大小为 0
            kernel_size = int(6 * sigma) | 1 # 确保为奇数
            blurred = cv2.GaussianBlur(base_img, (kernel_size, kernel_size), sigma)
        else:
            blurred = base_img.copy()
            
        return torch.from_numpy(blurred).float()