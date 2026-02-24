import subprocess
import time
import sys
import os

PYTHON_EXEC = sys.executable 
TARGET_SCRIPT = "main2.py"
TEST_SCRIPT = "test_ablation111.py"  
GPU_ID = "0" 

# 统一使用咱们最强的黄金参数
BASE_PARAMS = [
    "--gpu", GPU_ID, "--epochs", "200", "--batch_size", "2",
    "--lambda_rank", "0.2", "--lambda_ssl", "0.1", "--lambda_sub", "0.05",
    "--no_decoupling" # 保持解耦关闭
]

experiments = [
    # 实验1：真正的 Proposed 完全体 (修复Bug后，训练和测试全程启用拉普拉斯)
    {
        "exp_name": "Exp_v12_With_Laplacian_Fixed",
        "flags": [] 
    },
    # 实验2：消融拉普拉斯 (两边都直接输入RGB原图)
    {
        "exp_name": "Exp_v12_Without_Laplacian",
        "flags": ["--no_laplacian"] 
    }
]

def main():
    print(f"🚀 开始执行拉普拉斯先验 (Laplacian Prior) 消融实验...")
    for i, exp in enumerate(experiments):
        exp_name = exp["exp_name"]
        print(f"\n▶️ 正在运行: {exp_name}")
        
        train_cmd = [PYTHON_EXEC, TARGET_SCRIPT, "--exp_name", exp_name] + BASE_PARAMS
        if "flags" in exp: train_cmd.extend(exp["flags"])

        start_time = time.time()
        try:
            subprocess.run(train_cmd, check=True)
            print(f"✅ 训练完成！耗时: {(time.time() - start_time) / 60:.2f} 分钟")
            
            run_dir = os.path.join("results", exp_name)
            test_cmd = [PYTHON_EXEC, TEST_SCRIPT, "--run_dir", run_dir, "--gpu", GPU_ID]
            test_cmd.extend(["--no_decoupling"])
            if "flags" in exp: test_cmd.extend(exp["flags"])
                
            subprocess.run(test_cmd, check=True)
            print(f"✅ {exp_name} 测试完成！")
        except subprocess.CalledProcessError as e:
            print(f"❌ 实验 {exp_name} 失败！")

if __name__ == "__main__":
    main()