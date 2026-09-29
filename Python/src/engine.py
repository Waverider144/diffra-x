import torch
import torch.nn as nn
import json
import copy
import math
# 假设你把上一轮的 HopkinsOperator 类存放在了名为 hopkins_solver.py 的文件中
from propagator import HopkinsOperator 

class DiffractiveOpticalElement(nn.Module):
    def __init__(self, config_dict: dict, shape=(256, 256), device=torch.device('cpu')):
        super().__init__()
        self.device = device
        self.shape = shape
        self.config_dict = copy.deepcopy(config_dict)
        
        # 优化变量：透射元件的相位延迟 (初始化为 0 附近的小随机值)
        self.phase = nn.Parameter(torch.randn(shape, device=device) * 0.01)
        
        # 提取网格尺寸
        self.dx = config_dict["pixel_size"]
        
        # 计算 Padding 后的尺寸：256x256 补零后变成 512x512
        h, w = shape
        self.pad_h, self.pad_w = h // 4, w // 4  # 左右各补 w//2，总尺寸翻倍
        self.padded_shape = (h + 2 * self.pad_h, w + 2 * self.pad_w)
        
        self.source_params = copy.deepcopy(config_dict.get("source", {}))
        self.source_param_names = self._discover_optimizable_source_params()
        self._rebuild_hopkins_solver()

    def _discover_optimizable_source_params(self):
        """从 config['source'] 中找出可连续优化的光源参数。"""
        default_names = ["sigma_in", "sigma_out", "open_angle", "offset_angle"]
        configured = self.config_dict.get("source_optimize_params")
        candidate_names = configured if configured is not None else default_names
        return [
            name for name in candidate_names
            if isinstance(self.source_params.get(name), (int, float))
        ]

    def _source_bounds(self, name):
        num_poles = max(1, int(self.source_params.get("num_poles", 1)))
        bounds = self.config_dict.get("source_bounds", {})
        if name in bounds:
            return float(bounds[name][0]), float(bounds[name][1])
        if name == "sigma_in":
            return 0.0, 0.98
        if name == "sigma_out":
            return 0.01, 1.0
        if name == "open_angle":
            return 1.0, 360.0 / num_poles
        if name == "offset_angle":
            return 0.0, 360.0
        return -math.inf, math.inf

    def _clamp_source_params(self):
        for name in self.source_param_names:
            low, high = self._source_bounds(name)
            self.source_params[name] = min(max(float(self.source_params[name]), low), high)

        if "sigma_in" in self.source_params and "sigma_out" in self.source_params:
            sigma_in = float(self.source_params["sigma_in"])
            sigma_out = float(self.source_params["sigma_out"])
            min_gap = float(self.config_dict.get("source_sigma_min_gap", 0.01))
            sigma_out = max(sigma_out, sigma_in + min_gap)
            sigma_out = min(sigma_out, self._source_bounds("sigma_out")[1])
            sigma_in = min(sigma_in, sigma_out - min_gap)
            self.source_params["sigma_in"] = max(sigma_in, self._source_bounds("sigma_in")[0])
            self.source_params["sigma_out"] = sigma_out

    def _rebuild_hopkins_solver(self):
        """光源参数改变后重新计算 TCC/SOCS 模式。"""
        self._clamp_source_params()
        solver_config = copy.deepcopy(self.config_dict)
        solver_config["source"] = copy.deepcopy(self.source_params)
        json_config_str = json.dumps(solver_config)
        self.hopkins_solver = HopkinsOperator(
            json_config_str=json_config_str,
            shape=self.padded_shape,
            dx=self.dx,
            device=self.device
        )

    def get_source_parameters(self):
        return {name: float(self.source_params[name]) for name in self.source_param_names}

    def set_source_parameters(self, source_params, rebuild=True):
        for name, value in source_params.items():
            if name in self.source_param_names:
                self.source_params[name] = float(value)
        if rebuild:
            self._rebuild_hopkins_solver()

    def add_source_update(self, source_delta):
        for name, delta in source_delta.items():
            if name in self.source_param_names:
                self.source_params[name] = float(self.source_params[name]) + float(delta)
        self._rebuild_hopkins_solver()

    def forward(self) -> torch.Tensor:
        # 1. 计算紧贴 DOE 后的光场
        u_doe = torch.exp(1j * self.phase)
        
        # 2. 对光场进行补零扩展 (Padding)
        # (pad_left, pad_right, pad_top, pad_bottom)
        u_doe_padded = torch.nn.functional.pad(
            u_doe, 
            (self.pad_w, self.pad_w, self.pad_h, self.pad_h), 
            mode='constant', 
            value=0
        )
        
        # 3. 在大窗口下直接调用 Hopkins 算子计算部分相干光强
        # 算子内部会遍历相干模式，并自动调用你原有的 propagator 模块
        intensity_padded = self.hopkins_solver.compute_intensity(u_doe_padded)
        
        # 4. 传播完后，裁剪回中心的 256x256
        h, w = self.shape
        intensity = intensity_padded[self.pad_h : self.pad_h + h, self.pad_w : self.pad_w + w]
        
        # 5. 返回观测强度
        return intensity

def compute_l2_loss(output_intensity: torch.Tensor, target_intensity: torch.Tensor) -> torch.Tensor:
    """计算归一化 L2 范数损失 (MSE)"""
    return torch.mean((output_intensity - target_intensity) ** 2)

def _normalize_intensity(intensity: torch.Tensor) -> torch.Tensor:
    return intensity / torch.clamp(torch.max(intensity), min=1e-12)

def evaluate_normalized_loss(model: DiffractiveOpticalElement, target: torch.Tensor) -> torch.Tensor:
    return compute_l2_loss(_normalize_intensity(model()), target)

def _target_to_device(target_np, device):
    if isinstance(target_np, torch.Tensor):
        return target_np.to(device)
    return torch.tensor(target_np, dtype=torch.float32, device=device)

def _finite_difference_source_grad(model, target, base_loss, eps_map=None):
    """用中心差分估计光源参数梯度；Hopkins 模式会随参数重建。"""
    if eps_map is None:
        eps_map = {}

    original_params = model.get_source_parameters()
    gradients = {}

    for name, value in original_params.items():
        eps = float(eps_map.get(name, 0.5 if "angle" in name else 0.005))
        low, high = model._source_bounds(name)

        plus_params = dict(original_params)
        minus_params = dict(original_params)
        plus_params[name] = min(value + eps, high)
        minus_params[name] = max(value - eps, low)

        if abs(plus_params[name] - minus_params[name]) < 1e-12:
            gradients[name] = 0.0
            continue

        with torch.no_grad():
            model.set_source_parameters(plus_params, rebuild=True)
            plus_loss = float(evaluate_normalized_loss(model, target).item())
            model.set_source_parameters(minus_params, rebuild=True)
            minus_loss = float(evaluate_normalized_loss(model, target).item())

        gradients[name] = (plus_loss - minus_loss) / (plus_params[name] - minus_params[name])

    model.set_source_parameters(original_params, rebuild=True)
    return gradients

def optimize_mask_then_joint(model, annealer, base_target, config):
    """
    版本 1：
    先只优化 DOE 掩模 200 步，再进行 50 步掩模-光源联合优化。
    光源参数来自 config['source']，通过有限差分更新并重建 Hopkins/SOCS。
    """
    mask_steps = int(config.get("mask_pretrain_steps", 200))
    joint_steps = int(config.get("joint_steps", 50))
    mask_lr = float(config.get("learning_rate", 0.05))
    source_lr = float(config.get("source_learning_rate", 0.02))
    source_update_interval = int(config.get("source_update_interval", 1))
    total_steps = mask_steps + joint_steps

    optimizer = torch.optim.Adam([model.phase], lr=mask_lr)

    for step in range(total_steps):
        optimizer.zero_grad()
        target = _target_to_device(annealer.get_target_at_step(base_target, step), model.device)
        loss = evaluate_normalized_loss(model, target)
        loss.backward()
        optimizer.step()

        in_joint_stage = step >= mask_steps
        if in_joint_stage and model.source_param_names and ((step - mask_steps) % source_update_interval == 0):
            grad = _finite_difference_source_grad(model, target, float(loss.item()), config.get("source_fd_eps"))
            model.add_source_update({name: -source_lr * value for name, value in grad.items()})

        if step % 20 == 0:
            stage = "Joint" if in_joint_stage else "Mask"
            print(f"{stage} Step [{step:03d}/{total_steps}], Loss: {loss.item():.6f}, Source: {model.get_source_parameters()}")

    return model

def optimize_mask_then_conjugate_gradient(model, annealer, base_target, config):
    """
    版本 2：
    先只优化 DOE 掩模 200 步，再开启非线性共轭梯度做掩模-光源联合优化。
    相位梯度由 autograd 给出，光源梯度由中心差分给出。
    """
    mask_steps = int(config.get("mask_pretrain_steps", 200))
    cg_steps = int(config.get("cg_joint_steps", config.get("joint_steps", 50)))
    mask_lr = float(config.get("learning_rate", 0.05))
    cg_lr = float(config.get("cg_learning_rate", config.get("learning_rate", 0.05)))
    source_lr = float(config.get("source_cg_learning_rate", config.get("source_learning_rate", 0.02)))

    optimizer = torch.optim.Adam([model.phase], lr=mask_lr)

    for step in range(mask_steps):
        optimizer.zero_grad()
        target = _target_to_device(annealer.get_target_at_step(base_target, step), model.device)
        loss = evaluate_normalized_loss(model, target)
        loss.backward()
        optimizer.step()

        if step % 20 == 0:
            print(f"Mask Step [{step:03d}/{mask_steps + cg_steps}], Loss: {loss.item():.6f}")

    prev_phase_grad = None
    prev_source_grad = None
    phase_direction = None
    source_direction = None

    for cg_step in range(cg_steps):
        global_step = mask_steps + cg_step
        target = _target_to_device(annealer.get_target_at_step(base_target, global_step), model.device)

        if model.phase.grad is not None:
            model.phase.grad.zero_()
        loss = evaluate_normalized_loss(model, target)
        loss.backward()
        phase_grad = model.phase.grad.detach().clone()
        source_grad = _finite_difference_source_grad(model, target, float(loss.item()), config.get("source_fd_eps"))

        if prev_phase_grad is None:
            beta = 0.0
        else:
            numerator = torch.sum(phase_grad * (phase_grad - prev_phase_grad)).item()
            denominator = torch.sum(prev_phase_grad * prev_phase_grad).item()
            numerator += sum(
                source_grad[name] * (source_grad[name] - prev_source_grad.get(name, 0.0))
                for name in source_grad
            )
            denominator += sum(value * value for value in prev_source_grad.values())
            beta = max(0.0, numerator / max(denominator, 1e-12))

        phase_direction = -phase_grad if phase_direction is None else -phase_grad + beta * phase_direction
        source_direction = {
            name: -source_grad[name] + beta * (source_direction.get(name, 0.0) if source_direction else 0.0)
            for name in source_grad
        }

        with torch.no_grad():
            model.phase.add_(cg_lr * phase_direction)
        model.add_source_update({name: source_lr * value for name, value in source_direction.items()})

        prev_phase_grad = phase_grad
        prev_source_grad = source_grad

        if global_step % 20 == 0:
            print(f"CG Joint Step [{global_step:03d}/{mask_steps + cg_steps}], Loss: {loss.item():.6f}, Source: {model.get_source_parameters()}")

    return model

def run_configured_optimization(model, annealer, base_target, config):
    version = int(config.get("joint_optimization_version", config.get("source_optimization_version", 1)))
    if version == 2:
        return optimize_mask_then_conjugate_gradient(model, annealer, base_target, config)
    return optimize_mask_then_joint(model, annealer, base_target, config)
