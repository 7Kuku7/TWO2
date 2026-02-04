# test_ablation3.py 2026.2.4改修改了模块
# python test_ablation3.py --model_path results/Exp_v4/seed_3407/run_20260204_111027/best_model.pth
# python test_ablation3.py --model_path results/ka1/Ablation_wo_SSL/best_model.pth
# python test_ablation3.py --model_path results/ka1/Ablation_wo_MultiTask/best_model.pth --no_multitask

# test_ablation.py
import torch
from torch.utils.data import DataLoader
import numpy as np
import random
import json
import os
import argparse
import torchvision.transforms as T
from tqdm import tqdm

# ================= 导入模块 =================
from config import Config
from datasets.nerf_loader import NerfDataset
from models.dis_nerf_advanced import DisNeRFQA_Advanced
from utils import calculate_srcc, calculate_plcc, calculate_krcc

def parse_args():
    parser = argparse.ArgumentParser(description="Test Ablation Models (Direct Path)")
    
    # [修改] 改为直接接收模型路径
    parser.add_argument("--model_path", type=str, required=True, help="Full path to the model checkpoint (.pth)")
    parser.add_argument("--gpu", type=str, default="0")
    
    # 结构开关 (必须与训练时保持一致！)
    parser.add_argument("--no_fusion", action='store_true', help="Disable Adaptive Fusion")
    parser.add_argument("--no_multitask", action='store_true', help="Remove Sub-score Head")
    parser.add_argument("--no_decoupling", action='store_true', help="Remove MI Estimator")
    
    return parser.parse_args()

def main():
    args = parse_args()
    cfg = Config()
    
    # 1. 设置环境
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 结构开关解析
    use_fusion = not args.no_fusion
    use_multitask = not args.no_multitask
    use_decoupling = not args.no_decoupling
    
    # [新增] 解析输出目录 (基于模型路径)
    if not os.path.exists(args.model_path):
        print(f"Error: Model file not found at {args.model_path}")
        return
    
    output_dir = os.path.dirname(args.model_path)
    model_name = os.path.basename(args.model_path)

    print("="*60)
    print(f"Start Testing Model: {model_name}")
    print(f"  > Path: {args.model_path}")
    print(f"  > Output Dir: {output_dir}")
    print(f"  > Structure: Fusion={use_fusion}, MultiTask={use_multitask}, Decouple={use_decoupling}")
    print("="*60)

    # 2. 准备数据 (Test Set)
    basic_transform = T.Compose([
        T.Resize((224, 224)), 
        T.ToTensor(), 
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    test_set = NerfDataset(
        root_dir=cfg.ROOT_DIR,
        mos_file=cfg.MOS_FILE,
        mode='test',            
        basic_transform=basic_transform,
        ssl_transform=None,
        distortion_sampling=True, # 保持与训练一致 (拉普拉斯)
        num_frames=8,
        use_subscores=cfg.USE_SUBSCORES
    )
    
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False, num_workers=4)
    print(f"Test Set Loaded: {len(test_set)} samples.")

    # 3. 初始化模型
    model = DisNeRFQA_Advanced(
        num_subscores=4, 
        use_fusion=use_fusion, 
        use_multitask=use_multitask, 
        use_decoupling=use_decoupling
    )
    model.to(device)
    model.eval()

    # 4. 加载权重
    print(f"Loading weights...")
    try:
        checkpoint = torch.load(args.model_path, map_location=device)
        
        # 处理可能的 state_dict 键值前缀
        state_dict = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('module.'):
                new_state_dict[k[7:]] = v
            else:
                new_state_dict[k] = v
                
        model.load_state_dict(new_state_dict, strict=True)
        print("Weights loaded successfully.")
    except Exception as e:
        print(f"Weight loading failed! \n{e}")
        return

    # 5. 推理循环
    preds, targets, keys = [], [], []
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing"):
            x_c, x_d, score, _, key, _, _ = batch
            x_c, x_d = x_c.to(device), x_d.to(device)
            
            # Forward
            pred_score, _, _, _, _, _ = model(x_c, x_d)
            
            preds.extend(pred_score.cpu().numpy().flatten())
            targets.extend(score.numpy().flatten())
            keys.extend(key)

    # 6. 计算指标
    preds = np.array(preds)
    targets = np.array(targets)
    
    metrics = {
        "SRCC": float(calculate_srcc(preds, targets)),
        "PLCC": float(calculate_plcc(preds, targets)),
        "KRCC": float(calculate_krcc(preds, targets)),
        "RMSE": float(np.sqrt(np.mean((preds - targets)**2)))
    }
    
    print("\n" + "="*30)
    print(f"Test Results:")
    print(f"SRCC: {metrics['SRCC']:.4f}")
    print(f"PLCC: {metrics['PLCC']:.4f}")
    print(f"KRCC: {metrics['KRCC']:.4f}")
    print(f"RMSE: {metrics['RMSE']:.4f}")
    print("="*30 + "\n")
    
    # 保存结果到模型所在目录
    save_name = f"test_metrics_{model_name}.json" 
    res_path = os.path.join(output_dir, save_name)
    
    # 这里的 preds.tolist() 和 targets.tolist() 是安全的，它们会自动转换类型
    with open(res_path, 'w') as f:
        json.dump({
            "model_path": args.model_path,
            "metrics": metrics,       # <--- 现在这里面全是 python float 了，不会报错
            "preds": preds.tolist(),
            "targets": targets.tolist(),
            "keys": keys
        }, f, indent=4)
    print(f"Saved results to {res_path}")

if __name__ == "__main__":
    main()