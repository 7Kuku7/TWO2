import torch
import matplotlib.pyplot as plt
import os
import numpy as np
from models.dis_nerf_advanced4 import DisNeRFAdvanced
from config import Config
from datasets.nerf_loader import NerfDataset
from datasets.ssl_transforms import SelfSupervisedAugmentor
from torchvision import transforms

def main():
    cfg = Config()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = DisNeRFAdvanced(use_fusion=True, use_multitask=True).to(device)
    model.load_state_dict(torch.load("results/Exp_v13_RGB_Ours_Full/best_model.pth", map_location=device))
    model.eval()

    transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
    ssl_aug = SelfSupervisedAugmentor() # 使用你代码里的增强器
    test_set = NerfDataset(cfg.ROOT_DIR, cfg.MOS_FILE, mode='test', basic_transform=transform, distortion_sampling=False)

    num_samples = min(20, len(test_set))
    clean_scores, aug_scores = [], []

    with torch.no_grad():
        for i in range(num_samples):
            img_clean, _, _ = test_set[i]
            # 施加 SSL 增强 (将张量转回 PIL 再增强，再转回张量)
            img_pil = transforms.ToPILImage()(img_clean)
            img_aug = transform(ssl_aug(img_pil))

            img_clean = img_clean.unsqueeze(0).to(device)
            img_aug = img_aug.unsqueeze(0).to(device)

            score_clean, _, _, _, _ = model(img_clean, img_clean)
            score_aug, _, _, _, _ = model(img_aug, img_aug)

            clean_scores.append(score_clean.item())
            aug_scores.append(score_aug.item())

    # 画图
    plt.figure(figsize=(10, 6))
    for i in range(num_samples):
        plt.plot([i, i], [clean_scores[i], aug_scores[i]], color='gray', alpha=0.5)
        plt.scatter(i, clean_scores[i], color='blue', label='Clean' if i==0 else "")
        plt.scatter(i, aug_scores[i], color='red', label='Augmented (Degraded)' if i==0 else "")

    plt.title("SSL Proxy Ranking Constraint Effect")
    plt.ylabel("Predicted Score")
    plt.xlabel("Sample Index")
    plt.legend()
    plt.savefig("vis3_ssl_margin.png", dpi=300)
    print("✅ SSL 保序可视化已保存至 vis3_ssl_margin.png")

if __name__ == "__main__":
    main()
