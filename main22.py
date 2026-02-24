# 消融实验 接受结构化参数 命令行参数版
import torch
from torch.utils.data import DataLoader
import numpy as np
import random
import json
import os
import argparse
import torchvision.transforms as T
from config import Config
from core.solver3 import Solver
from datasets.nerf_loader import NerfDataset
from datasets.ssl_transforms import SelfSupervisedAugmentor 
from models.dis_nerf_advanced4 import DisNeRFQA_Advanced

class MultiScaleCrop:
    def __init__(self, size=224): self.size = size
    def __call__(self, img):
        scale = int(np.random.choice([224, 256, 288]))
        img = T.Resize(scale)(img)
        img = T.RandomCrop(self.size)(img)
        return img

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_name", type=str, default="Default")
    parser.add_argument("--gpu", type=str, default="0")
    parser.add_argument("--epochs", type=int, default=50)
    
    # [新增] 显存控制参数
    parser.add_argument("--batch_size", type=int, default=-1, help="Override batch size")
    parser.add_argument("--num_frames", type=int, default=-1, help="Override num frames")

    # Loss 权重 (Parameter Ablation)
    parser.add_argument("--lambda_ssl", type=float, default=-1)
    parser.add_argument("--lambda_mi", type=float, default=-1)
    parser.add_argument("--lambda_sub", type=float, default=-1)
    parser.add_argument("--lambda_rank", type=float, default=-1)
    
    # 结构开关 (Structural Ablation)
    parser.add_argument("--no_fusion", action='store_true', help="Disable Adaptive Fusion (Use Concat)")
    parser.add_argument("--no_multitask", action='store_true', help="Remove Sub-score Head")
    parser.add_argument("--no_decoupling", action='store_true', help="Remove MI Estimator")
    
    return parser.parse_args()

def main():
    # 强制清理显存
    torch.cuda.empty_cache()
    
    cfg = Config()
    args = parse_args()
    
    cfg.SEED = 3407

    # 应用参数
    cfg.EXP_NAME = args.exp_name
    cfg.GPU_ID = args.gpu
    cfg.EPOCHS = args.epochs
    
    # [显存相关] 覆盖参数
    if args.batch_size > 0:
        cfg.BATCH_SIZE = args.batch_size
    
    if args.num_frames > 0:
        cfg.NUM_FRAMES = args.num_frames

    # Loss 覆盖
    if args.lambda_ssl >= 0: cfg.LAMBDA_SSL = args.lambda_ssl
    if args.lambda_mi >= 0: cfg.LAMBDA_MI = args.lambda_mi
    if args.lambda_sub >= 0: cfg.LAMBDA_SUB = args.lambda_sub
    if args.lambda_rank >= 0: cfg.LAMBDA_RANK = args.lambda_rank
    
    # 结构覆盖
    use_fusion = not args.no_fusion
    use_multitask = not args.no_multitask
    use_decoupling = not args.no_decoupling

    # 联动逻辑：如果结构被移除，Loss 必须归零
    if not use_multitask: cfg.LAMBDA_SUB = 0.0
    if not use_decoupling: cfg.LAMBDA_MI = 0.0
    
    # 路径处理
    output_dir = os.path.join(cfg.OUTPUT_DIR, cfg.EXP_NAME)
    os.makedirs(output_dir, exist_ok=True)
    
    # Transforms
    basic_transform = T.Compose([
        MultiScaleCrop(224), T.ToTensor(), T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # SSL 模块
    ssl_augmentor = SelfSupervisedAugmentor() if cfg.LAMBDA_SSL > 0 else None

    # Dataset (使用关键字参数防止错位)
    train_set = NerfDataset(
        root_dir=cfg.ROOT_DIR,
        mos_file=cfg.MOS_FILE,
        mode='train',
        basic_transform=basic_transform,
        ssl_transform=ssl_augmentor,
        distortion_sampling=True,
        num_frames=cfg.NUM_FRAMES, 
        use_subscores=cfg.USE_SUBSCORES
    )
    
    val_set = NerfDataset(
        root_dir=cfg.ROOT_DIR,
        mos_file=cfg.MOS_FILE,
        mode='val',
        basic_transform=basic_transform,
        ssl_transform=None,
        distortion_sampling=False,
        num_frames=cfg.NUM_FRAMES,
        use_subscores=cfg.USE_SUBSCORES
    )
    
    print(f"Dataset: {len(train_set)} Train, {len(val_set)} Val. Batch Size: {cfg.BATCH_SIZE}, Frames: {cfg.NUM_FRAMES}")

    train_loader = DataLoader(train_set, cfg.BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_set, cfg.BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

    # Model
    print(f"Build Model | Fusion: {use_fusion} | MultiTask: {use_multitask} | Decouple: {use_decoupling}")
    
    model = DisNeRFQA_Advanced(
        num_subscores=4, 
        use_fusion=use_fusion, 
        use_multitask=use_multitask, 
        use_decoupling=use_decoupling
    )

    solver = Solver(model, cfg, train_loader, val_loader)

    best_combined_score = -1.0 # 初始化最佳综合分数 (SRCC + PLCC)
    
    for epoch in range(1, cfg.EPOCHS + 1):
        loss = solver.train_epoch(epoch)
        metrics, _, _, _ = solver.evaluate()
        
        # 提取当前的 SRCC 和 PLCC
        current_srcc = metrics['srcc']
        current_plcc = metrics['plcc']
        current_score = current_srcc + current_plcc # 计算总和
        
        print(f"Ep {epoch} | Loss: {loss:.4f} | Val SRCC: {current_srcc:.4f} | Val PLCC: {current_plcc:.4f} | Sum: {current_score:.4f}")
        
        # 判断综合分数是否超越了历史最佳
        if current_score > best_combined_score:
            best_combined_score = current_score
            print(f"  >>> [New Best] 综合分数提升至 {best_combined_score:.4f}，正在保存模型...")
            
            # [防报错机制] 将 numpy float32 转换为 python float，否则 json.dump 会报错
            safe_metrics = {k: float(v) for k, v in metrics.items()}
            
            solver.save_model(output_dir, epoch, safe_metrics)
            with open(os.path.join(output_dir, "best_results.json"), "w") as f:
                json.dump(safe_metrics, f, indent=4)

if __name__ == "__main__":
    main()