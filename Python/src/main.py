import json
import os
import torch
import numpy as np
import cv2

from image_loader import load_target_image
from annealing import TargetAnnealer
# 注意：确保这里引用的 DiffractiveOpticalElement 是我们上一轮修改后的版本
from engine import DiffractiveOpticalElement, run_configured_optimization

def load_config():
    config_path = "config.json" if os.path.exists("config.json") else "../config.json"
    config_dir = os.path.abspath(os.path.dirname(config_path))
    with open(config_path, "r") as f:
        config = json.load(f)
    if config.get("image_source") == "file" and not os.path.isabs(config.get("image_path", "")):
        config["image_path"] = os.path.join(config_dir, config["image_path"])
    return config, config_dir

def resolve_output_dir(config, config_dir):
    output_path = config.get("output", "./output")
    return output_path if os.path.isabs(output_path) else os.path.join(config_dir, output_path)

def main():
    # 1. 读取控制配置
    config, config_dir = load_config()
    output_dir = resolve_output_dir(config, config_dir)
    os.makedirs(output_dir, exist_ok=True)
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 2. 修改分辨率以适配配置
    # 物理分辨率改为 (256, 256)，因为你的 engine 默认设计尺寸以及优化参数都是基于该基准。
    # 如果要改成 (512, 512)，请同步修改 DiffractiveOpticalElement 的 shape 参数。
    shape = tuple(config.get("shape", [256, 256]))
    base_target = load_target_image(config, shape=shape)
    
    # 3. 初始化退火器与微分模型
    mask_steps = int(config.get("mask_pretrain_steps", 200))
    joint_version = int(config.get("joint_optimization_version", config.get("source_optimization_version", 1)))
    joint_steps = int(config.get("cg_joint_steps" if joint_version == 2 else "joint_steps", 50))
    total_iters = mask_steps + joint_steps
    annealer = TargetAnnealer(config, total_iters=total_iters)
    
    # 关键修正 ①：Hopkins 算子需要在一开始传入 config_dict 从而在初始化阶段计算一次 TCC。
    # 这样避免了在 forward 中重复进行耗时的 SVD 分解。
    doe_model = DiffractiveOpticalElement(config_dict=config, shape=shape, device=device).to(device)
    
    # 关键修正 ②：移除了对 u_in_amplitude (平面波) 的显式传入。
    # 因为在部分相干的 Hopkins 理论中，光源的非相干空间交叠已经由算子内部的 TCC (或 SOCS 模式) 全权接管。
    # 输入物变成了一个纯粹的复振幅调制元件（即 DOE 产生的相位面）。

    print(f"开始逆向设计优化... 联合优化版本: {joint_version}, 总步数: {total_iters}")
    run_configured_optimization(doe_model, annealer, base_target, config)

    # 4. 保存设计结果与重建图像
    doe_model.eval() # 切换到评估模式
    with torch.no_grad():
        # 再次运行前向传播获取最终的成像面强度
        final_intensity = doe_model()
        # 归一化到 [0, 1] 范围
        final_intensity_norm = final_intensity / torch.max(final_intensity)
        # 转换回 NumPy 矩阵
        reconstructed_img_np = final_intensity_norm.cpu().numpy()

    # 转换成 8位 无符号整数图像 (0-255) 以便保存
    reconstructed_img = (reconstructed_img_np * 255).astype(np.uint8)
    cv2.imwrite(os.path.join(output_dir, "reconstructed_image.png"), reconstructed_img)
    print("重建图像已保存至 'reconstructed_image.png'。")

    # 保存相位图
    final_phase = doe_model.phase.detach().cpu().numpy()
    final_phase_wrapped = np.mod(final_phase, 2 * 2 * np.pi) # 原代码写了 2*np.pi
    phase_img = (np.mod(final_phase, 2 * np.pi) / (2 * np.pi) * 255).astype(np.uint8)
    cv2.imwrite(os.path.join(output_dir, "optimized_doe_phase.png"), phase_img)
    print("最终 DOE 相位图已保存至 'optimized_doe_phase.png'。")

    with open(os.path.join(output_dir, "optimized_source_params.json"), "w") as f:
        json.dump(doe_model.get_source_parameters(), f, indent=4)
    print("优化后的光源参数已保存至 'optimized_source_params.json'。")

    try:
        # 从模型的 hopkins_solver 中提取归一化后的光源矩阵
        source_img_np = doe_model.hopkins_solver.get_source_profile_np(True)
        
        # 保存为 png 图像
        cv2.imwrite(os.path.join(output_dir, "source_profile.png"), source_img_np)
        print("💡 [SMO预备] 实际计算的光源形状已成功保存至 'source_profile.png'。")
    except AttributeError:
        print("⚠️ 导出光源失败：请确认 `doe_model` 内部是否已正确实例化 `hopkins_solver`。")

if __name__ == "__main__":
    config, config_dir = load_config()
    output_dir = resolve_output_dir(config, config_dir)
    os.makedirs(output_dir, exist_ok=True)
    # 快速测试脚本（保持原样）
    test_config = {"image_source": "text", "target_text": "ODE"}
    test_img = load_target_image(test_config, (256, 256))
    
    # 保存出来肉眼检查
    cv2.imwrite(os.path.join(output_dir, "test_target.png"), (test_img * 255).astype(np.uint8))
    print("测试目标图已保存至 'test_target.png'，请检查文字是否居中可见。")
    main()
