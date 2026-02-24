import subprocess
import time
import sys
import os

PYTHON_EXEC = sys.executable 
TARGET_SCRIPT = "main2.py"
TEST_SCRIPT = "test_ablation111.py"  
GPU_ID = "0" 

# 黄金参数组合：删掉了 --seed 传参，且全局关闭拉普拉斯！
BASE_PARAMS = [
    "--gpu", GPU_ID, 
    "--epochs", "200", 
    "--batch_size", "2",
    "--lambda_rank", "0.2", 
    "--lambda_ssl", "0.1", 
    "--lambda_sub", "0.05",
    "--no_decoupling",          # 彻底关闭互信息解耦
    "--no_laplacian"            # ⚠️ 核心！彻底关闭拉普拉斯，采用纯 RGB 双流
]

experiments = [
    # 1. 纯 RGB 下的 Baseline (无 Fusion, 无 MultiTask)
    {
        "exp_name": "Exp_v13_RGB_Model_A_Baseline",
        "flags": ["--no_fusion", "--no_multitask"] 
    },
    
    # 2. 纯 RGB 下的 MultiTask Only
    {
        "exp_name": "Exp_v13_RGB_Model_B_MT_Only",
        "flags": ["--no_fusion"]
    },

    # 3. 纯 RGB 下的 Fusion Only
    {
        "exp_name": "Exp_v13_RGB_Model_C_Fusion_Only",
        "flags": ["--no_multitask"]
    },

    # 4. 纯 RGB 下的全量模型 (即你刚刚跑出 0.9471 的那个)
    {
        "exp_name": "Exp_v13_RGB_Ours_Full",
        "flags": []
    }
]

def main():
    print(f"🚀 开始执行【纯 RGB 双分支】架构消融实验 (Exp_v13)...")
    print("="*60)

    for i, exp in enumerate(experiments):
        exp_name = exp["exp_name"]
        print(f"\n▶️ [{i+1}/{len(experiments)}] 正在运行: {exp_name}")
        
        train_cmd = [PYTHON_EXEC, TARGET_SCRIPT, "--exp_name", exp_name] + BASE_PARAMS
        if "flags" in exp: train_cmd.extend(exp["flags"])

        start_time = time.time()
        try:
            subprocess.run(train_cmd, check=True)
            duration = (time.time() - start_time) / 60
            print(f"✅ {exp_name} 训练完成！耗时: {duration:.2f} 分钟")
            
            run_dir = os.path.join("results", exp_name)
            # 测试脚本这里也把 --seed 删掉了
            test_cmd = [PYTHON_EXEC, TEST_SCRIPT, "--run_dir", run_dir, "--gpu", GPU_ID]
            test_cmd.extend(["--no_decoupling", "--no_laplacian"]) # 测试也必须保持纯 RGB
            if "flags" in exp: test_cmd.extend(exp["flags"])
                
            subprocess.run(test_cmd, check=True)
            print(f"✅ {exp_name} 测试完成！")
        except subprocess.CalledProcessError as e:
            print(f"❌ 实验 {exp_name} 失败！")

if __name__ == "__main__":
    main()