# main.py
import torch
from torch.utils.data import DataLoader
import numpy as np
import random
import json
import os
import datetime
import shutil
import torchvision.transforms as T

# ================= 导入模块 =================
from config222 import Config
from core.solver3 import Solver
from datasets.nerf_loader import NerfDataset
from datasets.ssl_transforms import SelfSupervisedAugmentor 
from models.dis_nerf_advanced44 import DisNeRFQA_Advanced

class MultiScaleCrop:
    def __init__(self, size=224): self.size = size
    def __call__(self, img):
        scale = int(np.random.choice([224, 256, 288]))
        img = T.Resize(scale)(img)
        img = T.RandomCrop(self.size)(img)
        return img

def set_seed(seed):
    if seed is None: return
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"\n[System] Global Seed Set to: {seed}")

def run_training(cfg, seed, timestamp):
    """
    封装单次训练过程
    """
    # 1. 设置当前循环的随机种子
    set_seed(seed)
    
    # 2. 生成对应当前种子的输出目录
    # 注意：这里调用的是修改后的 get_output_path(seed)
    output_dir = os.path.join(cfg.get_output_path(seed), f"run_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)

    # 3. 备份配置文件 (只在第一个种子跑的时候备份一次即可，或者每次都备份)
    src_config = "config.py"
    dst_config = os.path.join(output_dir, "config.py")
    if os.path.exists(src_config):
        shutil.copy(src_config, dst_config)

    print("="*50)
    print(f"Start Experiment: {cfg.EXP_NAME} | Seed: {seed}")
    print(f"Output Dir: {output_dir}")
    print("="*50)

    # 4. Transforms
    basic_transform = T.Compose([
        MultiScaleCrop(224), 
        T.ToTensor(), 
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    ssl_augmentor = None
    if cfg.LAMBDA_SSL > 0:
        ssl_augmentor = SelfSupervisedAugmentor()

    # 5. Dataset
    train_set = NerfDataset(
        root_dir=cfg.ROOT_DIR,
        mos_file=cfg.MOS_FILE,
        mode='train',
        basic_transform=basic_transform,
        ssl_transform=ssl_augmentor,
        distortion_sampling=True,
        use_subscores=cfg.USE_SUBSCORES
    )
    
    val_set = NerfDataset(
        root_dir=cfg.ROOT_DIR,
        mos_file=cfg.MOS_FILE,
        mode='val',
        basic_transform=basic_transform,
        ssl_transform=None,
        distortion_sampling=False,
        use_subscores=cfg.USE_SUBSCORES
    )

    train_loader = DataLoader(train_set, batch_size=cfg.BATCH_SIZE, shuffle=True, num_workers=cfg.NUM_WORKERS)
    val_loader = DataLoader(val_set, batch_size=cfg.BATCH_SIZE, shuffle=False, num_workers=cfg.NUM_WORKERS)

    # 6. Model
    model = DisNeRFQA_Advanced(num_subscores=4, use_fusion=cfg.USE_FUSION)

    # 7. Solver
    solver = Solver(model, cfg, train_loader, val_loader)

    # 8. Train Loop
    best_srcc = -1.0
    best_plcc = -1.0
    best_combined_score = -1.0
    
    for epoch in range(1, cfg.EPOCHS + 1):
        loss = solver.train_epoch(epoch)
        metrics, preds, targets, keys = solver.evaluate()
        
        current_srcc = metrics['srcc']
        current_plcc = metrics['plcc']
        current_score = current_srcc + current_plcc
        
        print(f"Seed {seed} | Ep {epoch} | Loss: {loss:.4f} | SRCC: {current_srcc:.4f} | PLCC: {current_plcc:.4f}")
        
        if current_score > best_combined_score:
            best_combined_score = current_score
            best_srcc = current_srcc
            best_plcc = current_plcc
            
            # 保存最佳模型
            solver.save_model(output_dir, epoch, metrics)
            
            # 保存结果
            if cfg.SAVE_PER_VIDEO_RESULT:
                res_path = os.path.join(output_dir, "best_results.json")
                safe_metrics = {k: float(v) for k, v in metrics.items()}
                safe_metrics['best_combined_score'] = float(best_combined_score)
                with open(res_path, 'w') as f:
                    json.dump({
                        "run_info": {"epoch": epoch, "seed": seed, "time": timestamp},
                        "metrics": safe_metrics,
                        "preds": preds.tolist(),
                        "targets": targets.tolist(),
                        "keys": keys
                    }, f, indent=4)

    print(f"--> Seed {seed} Finished. Best SRCC: {best_srcc:.4f}")
    
    # 返回这次跑出来的最佳指标，方便主函数统计
    return best_srcc, best_plcc, best_combined_score

def main():
    cfg = Config()
    
    # 生成统一的时间戳，方便归类
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 记录所有种子的结果
    summary = []
    
    # [核心循环] 遍历 Config 里定义的种子列表
    # 如果 Config 里还是写的单个 SEED，这里做一个兼容处理
    seeds_to_run = cfg.SEEDS if hasattr(cfg, 'SEEDS') else [cfg.SEED]
    
    print(f"Total Seeds to Run: {seeds_to_run}")
    
    for seed in seeds_to_run:
        try:
            srcc, plcc, score = run_training(cfg, seed, timestamp)
            summary.append({
                "seed": seed,
                "srcc": srcc,
                "plcc": plcc,
                "score": score
            })
        except Exception as e:
            print(f"!! Error running seed {seed}: {e}")
            import traceback
            traceback.print_exc()
    
    # 打印最终统计报告
    print("\n" + "="*50)
    print("FINAL SUMMARY (Sorted by Combined Score)")
    print("="*50)
    
    # 按综合得分从高到低排序
    summary.sort(key=lambda x: x['score'], reverse=True)
    
    for res in summary:
        print(f"Seed {res['seed']}: SRCC={res['srcc']:.4f}, PLCC={res['plcc']:.4f} (Score={res['score']:.4f})")
    
    if summary:
        best_run = summary[0]
        print("="*50)
        print(f"🏆 CHAMPION SEED: {best_run['seed']} (SRCC: {best_run['srcc']:.4f})")
        print("="*50)

if __name__ == "__main__":
    main()