import json
import math
import numpy as np
import torch
from typing import Dict, Any
import warnings

def angular_spectrum_propagate(u_in: torch.Tensor, wl: float, dx: float, z: float) -> torch.Tensor:
    """ 原有的角谱传播算子 """
    num_y, num_x = u_in.shape
    
    fx = torch.fft.fftfreq(num_x, d=dx, device=u_in.device)
    fy = torch.fft.fftfreq(num_y, d=dx, device=u_in.device)
    fY, fX = torch.meshgrid(fy, fx, indexing='ij')
    
    f_sq = fX**2 + fY**2
    term = (1.0 / wl)**2 - f_sq
    
    # 使用 math.sqrt 代替 torch.sqrt
    fx_max = dx * num_x / (2 * wl * math.sqrt((dx * num_x)**2 + z**2))
    fy_max = dx * num_y / (2 * wl * math.sqrt((dx * num_y)**2 + z**2))
    
    # 构造频域掩膜
    propagation_mask = (term > 0) & (torch.abs(fX) < fx_max) & (torch.abs(fY) < fy_max)
    
    phase = 2 * np.pi * torch.sqrt(torch.clamp(term, min=0)) * z
    H = torch.where(propagation_mask, torch.exp(1j * phase), torch.tensor(0.0 + 0j, device=u_in.device))
    
    U_in = torch.fft.fft2(u_in)
    U_out = U_in * H
    u_out = torch.fft.ifft2(U_out)
    
    return u_out

class HopkinsOperator:
    def __init__(self, json_config_str: str, shape: tuple, dx: float, device: torch.device = torch.device('cpu')):
        """
        初始化 Hopkins 算子
        :param json_config_str: 包含光源和系统参数的 JSON 字符串
        :param shape: 图像网格大小 (num_y, num_x)
        :param dx: 空间采样间隔 (Grid size)
        """
        self.device = device
        self.num_y, self.num_x = shape
        self.dx = dx
        
        # 解析配置
        config = json.loads(json_config_str)
        self.wl = config["wavelength"]
        self.na = config["NA"]
        self.z = config["distance"]
        self.source_params = config["source"]
        self.num_modes = config.get("num_modes", 10) # 提取的前 N 个主要相干模式
        
        # 初始化频域坐标
        fx = torch.fft.fftfreq(self.num_x, d=self.dx, device=self.device)
        fy = torch.fft.fftfreq(self.num_y, d=self.dx, device=self.device)
        self.fY, self.fX = torch.meshgrid(fy, fx, indexing='ij')
        
        # 计算 TCC 并进行相干分解 (SOCS)
        self.modes, self.weights = self._compute_socs_modes()

    # def _generate_source_profile(self) -> torch.Tensor:
    #     """根据 JSON 参数生成光源的强度分布 (频域)"""
    #     f_res = torch.sqrt(self.fX**2 + self.fY**2)
    #     # 光学截止频率 f_NA = NA / lambda
    #     f_na = self.na / self.wl
        
    #     if self.source_type == "annular":
    #         # 环形光源
    #         sigma_in = self.source_params["sigma_in"]
    #         sigma_out = self.source_params["sigma_out"]
    #         r_in = sigma_in * f_na
    #         r_out = sigma_out * f_na
    #         source = ((f_res >= r_in) & (f_res <= r_out)).float()
            
    #     elif self.source_type == "dipole":
    #         # 1. 基础物理参数提取
    #         sigma_center = self.source_params["sigma_center"]
    #         sigma_radius = self.source_params["sigma_radius"]
    #         # 核心新增：从 self.source_params 提取开角（角度制，例如 40度），若无则默认给个 35 度
    #         open_angle_deg = self.source_params.get("open_angle", 35.0) 
    #         orientation = self.source_params.get("orientation", "X")
            
    #         r_c = sigma_center * f_na
    #         r_r = sigma_radius * f_na
            
    #         # 2. 计算每个网格点的极坐标 (距离 rho 和 角度 theta)
    #         rho = torch.sqrt(self.fX**2 + self.fY**2)
    #         theta = torch.atan2(self.fY, self.fX) # 返回范围 [-pi, pi]
            
    #         # 3. 约束条件一：限制在两个极点圆内 (保留你原有的局部圆约束)
    #         if orientation.upper() == "X":
    #             dist1 = torch.sqrt((self.fX - r_c)**2 + self.fY**2)
    #             dist2 = torch.sqrt((self.fX + r_c)**2 + self.fY**2)
    #         else:
    #             dist1 = torch.sqrt(self.fX**2 + (self.fY - r_c)**2)
    #             dist2 = torch.sqrt(self.fX**2 + (self.fY + r_c)**2)
    #         inside_circles = (dist1 <= r_r) | (dist2 <= r_r)
            
    #         # 4. 核心新增：约束条件二：限制在扇形开角范围内
    #         # 将角度转换为绝对值，方便关于主轴对称判断
    #         half_angle_rad = math.radians(open_angle_deg / 2.0)
            
    #         if orientation.upper() == "X":
    #             # X 方向 Dipole 镜像关于 0 弧度(右) 和 pi/-pi 弧度(左)
    #             # 统一映射到接近 0 和 pi 的差值
    #             angle_match = (torch.abs(theta) <= half_angle_rad) | (torch.abs(torch.abs(theta) - np.pi) <= half_angle_rad)
    #         else:
    #             # Y 方向 Dipole 镜像关于 pi/2(上) 和 -pi/2(下)
    #             angle_match = (torch.abs(theta - np.pi/2) <= half_angle_rad) | (torch.abs(theta + np.pi/2) <= half_angle_rad)
                
    #         # 5. 求交集得到标准的工业级 Sector Dipole
    #         source = (inside_circles & angle_match).float()
            
    #     else: # "conventional" 或默认常规光源
    #         sigma = self.source_params.get("sigma", 1.0)
    #         r_max = sigma * f_na
    #         source = (f_res <= r_max).float()

    #     # === 核心修正：只在无法解析的异常情况下触发报警与保底 ===
    #     if torch.sum(source) == 0:
    #         df_x = 1.0 / (self.num_x * self.dx)
    #         df_y = 1.0 / (self.num_y * self.dx)
            
    #         if self.source_type == "dipole":
    #             r_r = self.source_params["sigma_radius"] * f_na
    #             error_msg = (
    #                 f"\n❌ 【物理参数冲突错误】: 无法在当前频域网格中解析双极光源！\n"
    #                 f"原因分析:\n"
    #                 f"  - 你的双极单圆半径为: {r_r:.4e} m^-1\n"
    #                 f"  - 当前网格的分辨率(格点间距)为: df_x = {df_x:.4e} m^-1\n"
    #             )
    #         else:
    #             error_msg = f"\n❌ 无法解析光源频域掩膜。"

    #         warnings.warn(error_msg + "\n⚠️ [警告] 程序已强制点亮中心像素保底运行，结果可能严重失真！")
            
    #         # 强制点亮保底
    #         if self.source_type == "dipole":
    #             idx1 = torch.argmin(torch.sqrt((self.fX - r_c)**2 + self.fY**2))
    #             idx2 = torch.argmin(torch.sqrt((self.fX + r_c)**2 + self.fY**2))
    #             source.view(-1)[idx1] = 1.0
    #             source.view(-1)[idx2] = 1.0
    #         else:
    #             center_idx = torch.argmin(self.fX**2 + self.fY**2)
    #             source.view(-1)[center_idx] = 1.0
            
    #     # 归一化能量
    #     if torch.sum(source) > 0:
    #         source = source / torch.sum(source)
    #     return source
    
    def generate_parametric_source(self) -> torch.Tensor:
        """
        通用多极参数化光源生成器
        """
        f_na = self.na / self.wl
        sigma_in = self.source_params["sigma_in"] * f_na
        sigma_out = self.source_params["sigma_out"] * f_na
        open_angle_rad = math.radians(self.source_params["open_angle"])
        offset_rad = math.radians(self.source_params.get("offset_angle", 0.0))
        num_poles = self.source_params.get("num_poles", 2) # 2=Dipole, 4=Quasar, 1=环形/常规
        
        # 1. 半径约束 (环形切分)
        rho = torch.sqrt(self.fX**2 + self.fY**2)
        radial_mask = (rho >= sigma_in) & (rho <= sigma_out)
        
        if self.source_params["open_angle"] >= 360.0 / num_poles:
            # 如果开角足够大，直接退化为无角度限制的环形/传统光源
            return radial_mask.float()
            
        # 2. 角度约束 (多极切分)
        theta = torch.atan2(self.fY, self.fX) # [-pi, pi]
        
        # 将坐标轴旋转 offset_angle，并归一化到 [0, 2*pi]
        theta_swapped = torch.remainder(theta - offset_rad, 2 * np.pi)
        
        # 每个极点占据的周期角空间
        pole_period = 2 * np.pi / num_poles
        
        # 计算每个格点距离最近的极点中心轴的角度差
        local_theta = torch.remainder(theta_swapped + pole_period / 2, pole_period) - pole_period / 2
        
        # 扇形开角掩膜
        angular_mask = torch.abs(local_theta) <= (open_angle_rad / 2.0)
        
        return (radial_mask & angular_mask).float()

    def _compute_socs_modes(self):
        """计算交叉透射系数 (TCC) 并进行特征值分解 (SOCS)"""
        # 1. 生成完整、已归一化的光源频域采样分布
        source = self.generate_parametric_source()
        active_idx = torch.where(source > 0)
        s_y, s_x = active_idx[0], active_idx[1]
        num_source_pts = len(s_y)
        
        if num_source_pts == 0:
            raise ValueError("光源强度全为0，请检查参数设置。")
            
        # === 物理光源网格的稀疏化下采样 (加入能量补偿修正) ===
        MAX_SOURCE_POINTS = 256  # 工业界标准点数
        
        if num_source_pts > MAX_SOURCE_POINTS:
            # 计算跨步步长
            stride = math.ceil(num_source_pts / MAX_SOURCE_POINTS)
            
            # 执行下采样
            s_y_down = s_y[::stride]
            s_x_down = s_x[::stride]
            num_downsampled_pts = len(s_y_down)
            
            print(f"ℹ️ [性能优化] 检测到高密度光源(共 {num_source_pts} 个格点)。")
            print(f"    已自动启用等间隔下采样(Stride={stride})至 {num_downsampled_pts} 个源点进行 TCC 分解。")
            
            # 【核心修正】计算当前保留下来的格点的原始能量总和
            sampled_energy_sum = torch.sum(source[s_y_down, s_x_down])
            
            # 提取下采样点的强度，并强制通过比例缩放将其补回 1.0，实现能量归一化守恒
            # 这样即使点数变少了，光源的总辐射能量在物理上依然保持 100%
            source_points_intensity = source[s_y_down, s_x_down] / sampled_energy_sum
            
            # 更新指针和计数
            s_y, s_x = s_y_down, s_x_down
            num_source_pts = num_downsampled_pts
        else:
            # 如果不需要下采样，直接提取原强度即可
            source_points_intensity = source[s_y, s_x]
        # =======================================================

        # 2. 构造光学系统的物镜瞳函数 (Pupil function)
        f_res = torch.sqrt(self.fX**2 + self.fY**2)
        pupil = (f_res <= (self.na / self.wl)).to(torch.complex64)
        
        # 3. 构造低维度的特征矩阵
        A = torch.zeros((self.num_y * self.num_x, num_source_pts), dtype=torch.complex64, device=self.device)
        
        for i in range(num_source_pts):
            src_fy = self.fY[s_y[i], s_x[i]]
            src_fx = self.fX[s_y[i], s_x[i]]
            
            # 使用修正后严格能量守恒的电场振幅权重
            weight = torch.sqrt(source_points_intensity[i])
            
            # 频域平移后的瞳函数
            shifted_fX = self.fX + src_fx
            shifted_fY = self.fY + src_fy
            shifted_f_res = torch.sqrt(shifted_fX**2 + shifted_fY**2)
            shifted_pupil = (shifted_f_res <= (self.na / self.wl)).to(torch.complex64)
            
            # 存入矩阵列向量
            A[:, i] = (weight * shifted_pupil).flatten()
            
        # 4. 对 A 进行奇异值分解 (SVD)，求取 TCC 主模式
        U, S, _ = torch.linalg.svd(A, full_matrices=False)
        
        # 5. 截取前 N 个主相干模式
        n_m = min(self.num_modes, num_source_pts)
        weights = (S[:n_m] ** 2)
        
        modes = []
        for i in range(n_m):
            mode_f = U[:, i].reshape(self.num_y, self.num_x)
            modes.append(mode_f)
            
        return modes, weights

    def compute_intensity(self, mask_profile: torch.Tensor) -> torch.Tensor:
        """
        利用 Hopkins 算子（SOCS模式）计算像面的总强度
        :param mask_profile: 输入的掩膜/物函数矩阵 (空间域，复振幅)
        :return: 像面光强分布 (空间域，实数)
        """
        total_intensity = torch.zeros((self.num_y, self.num_x), dtype=torch.float32, device=self.device)
        mask_fft = torch.fft.fft2(mask_profile)
        
        for mode_f, weight in zip(self.modes, self.weights):
            # 1. 在频域与相干内核（TCC分解出的模式）做乘积
            # 根据 Hopkins 理论，这里是物方频谱与相干模式的频域卷积（等价于平移后的乘积）
            u_out_fft = mask_fft * mode_f
            u_out_space = torch.fft.ifft2(u_out_fft)
            
            # 2. 通过你原有的角谱算子进行长距离自由空间传播
            u_propagated = angular_spectrum_propagate(u_out_space, self.wl, self.dx, self.z)
            
            # 3. 模方叠加
            total_intensity += weight * (torch.abs(u_propagated) ** 2)
            
        return total_intensity
    
    def get_source_profile_np(self, show_downsampled: bool = False) -> np.ndarray:
        """
        获取用于导出的光源二维强度分布矩阵 (NumPy 格式，已归一化到 0~255 亮度)
        
        :param show_downsampled: True 代表只点亮实际参与 TCC 计算的离散下采样点 (debug 用)
                                False 代表导出完整的参数化连续光源几何形状
        """
        with torch.no_grad():
            # 1. 重新生成频域光源 profile
            source_tensor = self.generate_parametric_source().clone()
            
            # 2. 如果用户想要看下采样后的实际离散点
            if show_downsampled:
                active_idx = torch.where(source_tensor > 0)
                s_y, s_x = active_idx[0], active_idx[1]
                num_source_pts = len(s_y)
                
                MAX_SOURCE_POINTS = 256
                if num_source_pts > MAX_SOURCE_POINTS:
                    stride = math.ceil(num_source_pts / MAX_SOURCE_POINTS)
                    # 先把原图清空
                    downsampled_source = torch.zeros_like(source_tensor)
                    # 只把保留的下采样点设为 1.0
                    downsampled_source[s_y[::stride], s_x[::stride]] = 1.0
                    source_tensor = downsampled_source
            
            # 3. 将 Tensor 转回 CPU 上的 NumPy 数组
            source_np = source_tensor.detach().cpu().numpy()
            
            # 4. 健壮的归一化：由于 _generate_source_profile 内部做过能量归一化 (sum)，
            # 这里必须重新找最大值，将其拉满到 255 亮度的可视化图像
            max_val = np.max(source_np)
            if max_val > 0:
                source_img = (source_np / max_val * 255).astype(np.uint8)
            else:
                source_img = np.zeros_like(source_np, dtype=np.uint8)
            
            # 5. 做频域中心平移 (fftshift)，让低频/直流通量居中显示，符合光学查看习惯
            source_img_shifted = np.fft.fftshift(source_img)
            
            return source_img_shifted