# python test_ablation111.py --run_dir results/Exp_v9_Ours_Full/
# python test_ablation111.py --run_dir results/Exp_v9_wo_Fusion/ --no_fusion
# python test_ablation111.py --run_dir results/Exp_v9_wo_Decoupling/ --no_decoupling
# python test_ablation111.py --run_dir results/Exp_v9_wo_MultiTask/ --no_multitask

import torch
from torch.utils.data import DataLoader
import json
import os
import argparse
import torchvision.transforms as T
import numpy as np
import random
from config import Config
from datasets.nerf_loader import NerfDataset
from models.dis_nerf_advanced4 import DisNeRFQA_Advanced
from scipy.stats import pearsonr, spearmanr, kendalltau

class MultiScaleCrop:
    def __init__(self, size=224): self.size = size
    def __call__(self, img):
        scale = int(np.random.choice([224, 256, 288]))
        img = T.Resize(scale)(img)
        img = T.CenterCrop(self.size)(img) # 测试时通常用 CenterCrop 更稳定
        return img

def parse_args():
    parser = argparse.ArgumentParser(description="Test Ablation Models")
    parser.add_argument("--run_dir", type=str, required=True, help="Directory containing best_model.pth")
    parser.add_argument("--gpu", type=str, default="0")
    
    # 结构开关 (Structural Ablation) - 和 main2.py 保持一致
    parser.add_argument("--no_fusion", action='store_true', help="Disable Adaptive Fusion")
    parser.add_argument("--no_multitask", action='store_true', help="Remove Sub-score Head")
    parser.add_argument("--no_decoupling", action='store_true', help="Remove MI Estimator")

    parser.add_argument("--no_laplacian", action='store_true')
    
    return parser.parse_args()

def set_seed(seed=3407):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"[Test] Set Global Seed: {seed}")

def test():
    args = parse_args()
    cfg = Config()

    use_lap = not args.no_laplacian
    
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    set_seed(cfg.SEED)

    # 模型结构覆盖逻辑
    use_fusion = not args.no_fusion
    use_multitask = not args.no_multitask
    use_decoupling = not args.no_decoupling

    # 1. Dataset
    basic_transform = T.Compose([
        MultiScaleCrop(224), 
        T.ToTensor(), 
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    test_set = NerfDataset(
        root_dir=cfg.ROOT_DIR,
        mos_file=cfg.MOS_FILE,
        mode='test', # 确保是 test 模式
        basic_transform=basic_transform,
        ssl_transform=None,
        distortion_sampling=use_lap,
        num_frames=cfg.NUM_FRAMES,
        use_subscores=cfg.USE_SUBSCORES
    )
    
    test_loader = DataLoader(test_set, batch_size=cfg.BATCH_SIZE, shuffle=False, num_workers=4)
    print(f"[Test] Test Set Size: {len(test_set)}")

    # 2. Build Model
    print(f"[Test] Build Model | Fusion: {use_fusion} | MultiTask: {use_multitask} | Decouple: {use_decoupling}")
    model = DisNeRFQA_Advanced(
        num_subscores=4, 
        use_fusion=use_fusion, 
        use_multitask=use_multitask, 
        use_decoupling=use_decoupling
    ).to(device)

    # 3. Load Weights
    model_path = os.path.join(args.run_dir, "best_model.pth")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Cannot find model file at: {model_path}")
        
    print(f"[Test] Loading weights from: {model_path}")
    checkpoint = torch.load(model_path, map_location=device)
    
    # 兼容处理：如果你保存的 checkpoint 是整个字典还是单纯的 state_dict
    if 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
    else:
        model.load_state_dict(checkpoint)
        
    model.eval()

    # 4. Evaluation Loop
    all_preds = []
    all_targets = []
    all_keys = []

    print("[Test] Start inferencing...")
    with torch.no_grad():
        for batch in test_loader:
            # [关键修改] 使用索引解包从 Dataset 返回的 Tuple
            x_c = batch[0].to(device)     # content_input
            x_d = batch[1].to(device)     # distortion_input
            labels = batch[2].to(device).float() # score_tensor
            keys = batch[4]               # key

            # Forward
            score, _, _, _, _, _ = model(x_c, x_d)
            
            all_preds.extend(score.squeeze(-1).cpu().numpy().tolist())
            all_targets.extend(labels.cpu().numpy().tolist())
            all_keys.extend(keys)

    # 5. Calculate Metrics
    preds_np = np.array(all_preds)
    targets_np = np.array(all_targets)

    srcc, _ = spearmanr(preds_np, targets_np)
    plcc, _ = pearsonr(preds_np, targets_np)
    krcc, _ = kendalltau(preds_np, targets_np)
    rmse = np.sqrt(np.mean((preds_np - targets_np) ** 2))

    print("="*40)
    print(f"✅ Test Results for: {args.run_dir}")
    print(f"SRCC: {srcc:.4f}")
    print(f"PLCC: {plcc:.4f}")
    print(f"KRCC: {krcc:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print("="*40)

    # 6. Save test results
    results_dict = {
        "metrics": {
            "srcc": float(srcc),
            "plcc": float(plcc),
            "krcc": float(krcc),
            "rmse": float(rmse)
        },
        "preds": all_preds,
        "targets": all_targets,
        "keys": all_keys
    }
    
    save_path = os.path.join(args.run_dir, "test_results.json")
    with open(save_path, "w") as f:
        json.dump(results_dict, f, indent=4)
    print(f"Test results saved to {save_path}")

if __name__ == "__main__":
    test()