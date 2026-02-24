import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from models.dis_nerf_advanced4 import DisNeRFAdvanced
from config import Config
from datasets.nerf_loader import NerfDataset
from torchvision import transforms

def get_saliency(model, image_tensor):
    model.eval()
    # 开启梯度追踪
    image_tensor.requires_grad_()
    
    # 前向传播 (模拟纯RGB双流输入)
    score, _, _, _, _ = model(image_tensor, image_tensor)
    
    # 对预测分数反向传播，求输入的梯度
    model.zero_grad()
    score.backward()
    
    # 获取梯度绝对值并在通道维度求最大值
    saliency = image_tensor.grad.abs().max(dim=1)[0].squeeze().cpu().numpy()
    
    # 归一化到 0-255
    saliency = (saliency - saliency.min()) / (saliency.max() - saliency.min() + 1e-8)
    saliency = np.uint8(255 * saliency)
    return saliency

def main():
    cfg = Config()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # 加载模型
    model = DisNeRFAdvanced(use_fusion=True, use_multitask=True).to(device)
    model_path = "results/Exp_v13_RGB_Ours_Full/best_model.pth"
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        print("✅ 模型加载成功！")
    else:
        print("❌ 找不到模型权重，请先跑完 Exp_v13_RGB_Ours_Full")
        return

    # 加载一张测试集图片 (关闭拉普拉斯)
    transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
    test_set = NerfDataset(cfg.ROOT_DIR, cfg.MOS_FILE, mode='test', basic_transform=transform, distortion_sampling=False)
    
    img, _, _ = test_set[0] # 取第一张图
    img_input = img.unsqueeze(0).to(device)
    
    # 获取显著性图
    saliency_map = get_saliency(model, img_input)
    
    # 可视化原图和热力图
    img_vis = img.permute(1, 2, 0).cpu().numpy()
    heatmap = cv2.applyColorMap(saliency_map, cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    
    # 叠加
    overlay = 0.5 * img_vis + 0.5 * (heatmap / 255.0)
    
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.title("Original Image")
    plt.imshow(img_vis)
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.title("Network Saliency (Focus Area)")
    plt.imshow(overlay)
    plt.axis('off')
    plt.savefig("vis1_saliency.png", dpi=300)
    print("✅ 可视化结果已保存至 vis1_saliency.png")

if __name__ == "__main__":
    main()
