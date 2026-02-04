# test_ablation1.py
# python test_ablation1.py --run_dir results/Struct_Full_Model
# python test_ablation1.py --run_dir results/Struct_wo_Fusion --no_fusion
# python test_ablation1.py --run_dir results/Struct_wo_MultiTask --no_multitask
# python test_ablation1.py --run_dir results/Struct_wo_Decoupling --no_decoupling
# python test_ablation1.py --run_dir results/Struct_wo_SSL

import torch
from torch.utils.data import DataLoader
import torchvision.transforms as T
import numpy as np
import random
import os
import json
import argparse

# 引入项目模块 (确保这些文件都在项目根目录下或PYTHONPATH中)
from config import Config
from datasets.nerf_loader import NerfDataset
from models.dis_nerf_advanced import DisNeRFQA_Advanced
from utils import calculate_srcc, calculate_plcc, calculate_krcc

def parse_args():
    parser = argparse.ArgumentParser(description="Ablation Test Script")
    
    # 必填：你的实验结果文件夹路径 (包含 best_model.pth 的那个文件夹)
    parser.add_argument("--run_dir", type=str, required=True, help="Path to the specific run folder (e.g., results/Struct_wo_Fusion/run_xxx)")
    
    # === 关键：模块独立开关 ===
    # 如果你在训练时用了 --no_fusion，测试时必须加上 --no_fusion
    parser.add_argument("--no_fusion", action='store_true', help="Disable Fusion Module")
    parser.add_argument("--no_multitask", action='store_true', help="Disable Multi-task Head")
    parser.add_argument("--no_decoupling", action='store_true', help="Disable Decoupling Module")
    
    return parser.parse_args()

def test():
    args = parse_args()
    cfg = Config() # 加载默认配置
    
    # 1. 解析结构开关
    # 如果 args.no_fusion 为 True (命令行加了参数)，则 use_fusion 变成 False
    use_fusion = not args.no_fusion
    use_multitask = not args.no_multitask
    use_decoupling = not args.no_decoupling
    
    print("="*60)
    print(f"[Test] Run Directory: {args.run_dir}")
    print(f"[Test] Model Structure:")
    print(f"   - Fusion:      {'✅ ON' if use_fusion else '❌ OFF'}")
    print(f"   - Multi-task:  {'✅ ON' if use_multitask else '❌ OFF'}")
    print(f"   - Decoupling:  {'✅ ON' if use_decoupling else '❌ OFF'}")
    print("="*60)

    # 2. 准备测试数据
    # 保持与训练一致的预处理，但通常测试时只做 Resize 或 CenterCrop
    test_transform = T.Compose([
        T.Resize((224, 224)), 
        T.ToTensor(), 
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    test_set = NerfDataset(
        root_dir=cfg.ROOT_DIR,
        mos_file=cfg.MOS_FILE,
        mode='test', 
        basic_transform=test_transform,
        ssl_transform=None,
        distortion_sampling=False,
        use_subscores=cfg.USE_SUBSCORES 
    )
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False, num_workers=4)

    # 3. 初始化模型 (传入结构参数，实现物理隔离)
    device = torch.device(f"cuda:{cfg.GPU_ID}")
    
    try:
        model = DisNeRFQA_Advanced(
            num_subscores=4, 
            use_fusion=use_fusion,          # <--- 关键点
            use_multitask=use_multitask,    # <--- 关键点
            use_decoupling=use_decoupling   # <--- 关键点
        ).to(device)
    except Exception as e:
        print(f"Error initializing model: {e}")
        return
    
    # 4. 加载权重
    ckpt_path = os.path.join(args.run_dir, "best_model.pth")
    if not os.path.exists(ckpt_path):
        print(f"Error: Checkpoint not found at {ckpt_path}")
        return

    print(f"[Test] Loading weights...")
    checkpoint = torch.load(ckpt_path, map_location=device)
    
    try:
        # strict=True 确保权重文件和模型结构必须完美对应
        model.load_state_dict(checkpoint['state_dict'], strict=True)
        print("[Success] Weights loaded successfully.")
    except RuntimeError as e:
        print("\n" + "!"*60)
        print("[Error] 权重加载失败！这通常是因为测试命令和训练时的结构不匹配。")
        print("请检查是否忘记加 --no_fusion, --no_multitask 或 --no_decoupling 参数。")
        print("详细错误信息如下：")
        print(e)
        print("!"*60 + "\n")
        return

    model.eval()

    # 5. 推理循环
    preds, targets, keys = [], [], []
    with torch.no_grad():
        for batch in test_loader:
            x_c, x_d, score, _, key, _, _ = batch
            x_c, x_d = x_c.to(device), x_d.to(device)
            
            # 前向传播
            pred_score, _, _, _, _, _ = model(x_c, x_d)
            
            preds.extend(pred_score.cpu().numpy().flatten())
            targets.extend(score.numpy().flatten())
            keys.extend(key)

    # 6. 计算指标
    preds = np.array(preds)
    targets = np.array(targets)
    
    srcc = calculate_srcc(preds, targets)
    plcc = calculate_plcc(preds, targets)
    rmse = np.sqrt(np.mean((preds*100 - targets*100)**2))

    print(f"\n>>> Final Results: SRCC={srcc:.4f}, PLCC={plcc:.4f}, RMSE={rmse:.4f}")

    # 保存结果到对应目录
    save_path = os.path.join(args.run_dir, "test_results_manual.json")
    with open(save_path, "w") as f:
        # [修复点] 必须加 float() 转换，否则 json 报错
        json.dump({
            "srcc": float(srcc), 
            "plcc": float(plcc), 
            "rmse": float(rmse), 
            "preds": preds.tolist()
        }, f, indent=4)
    print(f"[Test] Results saved to {save_path}")

if __name__ == "__main__":
    test()