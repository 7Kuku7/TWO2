import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import os
from models.dis_nerf_advanced4 import DisNeRFAdvanced
from config import Config
from datasets.nerf_loader import NerfDataset
from torchvision import transforms
from torch.utils.data import DataLoader

def main():
    cfg = Config()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # 加载模型
    model = DisNeRFAdvanced(use_fusion=True, use_multitask=True).to(device)
    model.load_state_dict(torch.load("results/Exp_v13_RGB_Ours_Full/best_model.pth", map_location=device))
    model.eval()

    transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
    test_set = NerfDataset(cfg.ROOT_DIR, cfg.MOS_FILE, mode='test', basic_transform=transform, distortion_sampling=False)
    loader = DataLoader(test_set, batch_size=4, shuffle=False)

    features_list, mos_list = [], []

    with torch.no_grad():
        for img, mos, _ in loader:
            img = img.to(device)
            # 提取融合后的特征 (在 dis_nerf_advanced4.py 中，通常可以通过修改 forward 返回)
            # 这里我们利用模型的前向传播，截取 shared_features
            c_feat = model.content_encoder(img)
            d_feat = model.distortion_encoder(img)
            if hasattr(model, 'adaptive_fusion'):
                shared_feat, _ = model.adaptive_fusion(c_feat, d_feat)
            else:
                shared_feat = torch.cat((c_feat, d_feat), dim=1)
            
            features_list.append(shared_feat.cpu().numpy())
            mos_list.append(mos.numpy())

    features = np.concatenate(features_list, axis=0)
    labels = np.concatenate(mos_list, axis=0)

    # 运行 t-SNE
    tsne = TSNE(n_components=2, random_state=42)
    features_2d = tsne.fit_transform(features)

    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(features_2d[:, 0], features_2d[:, 1], c=labels, cmap='viridis', s=50)
    plt.colorbar(scatter, label="MOS Score")
    plt.title("t-SNE of Fused Features (Multi-task)")
    plt.savefig("vis2_tsne.png", dpi=300)
    print("✅ t-SNE 可视化结果已保存至 vis2_tsne.png")

if __name__ == "__main__":
    main()
