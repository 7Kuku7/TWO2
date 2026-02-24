import subprocess
import time
import sys
import os

# ================= 配置区域 =================
PYTHON_EXEC = sys.executable 
TARGET_SCRIPT = "main22.py"         # <--- 改用你新建的 main22.py
TEST_SCRIPT = "test_ablation111.py"  
GPU_ID = "1"                        # <--- 指定卡 1

# 基础参数 (不需要传 seed，因为你在 main22 里已经写死了)
BASE_PARAMS = [
    "--gpu", GPU_ID,
    "--epochs", "200",
    "--batch_size", "2",
    "--lambda_rank", "0.2",   
    "--lambda_ssl", "0.1",    
    "--lambda_sub", "0.05"
]

# ================= 实验计划 (隔离文件夹) =================
experiments = [
    {
        # [重要] 文件夹名字必须和卡0区分开
        "exp_name": "Exp_v11_Model_A_Baseline_s3407",
        "flags": ["--no_fusion", "--no_multitask", "--no_decoupling"] 
    },
    {
        "exp_name": "Exp_v11_Model_B_MT_Only_s3407",
        "flags": ["--no_fusion", "--no_decoupling"]
    },
    {
        "exp_name": "Exp_v11_Model_C_Fusion_Only_s3407",
        "flags": ["--no_multitask", "--no_decoupling"]
    },
    {
        "exp_name": "Exp_v11_Ours_Proposed_s3407",
        "flags": ["--no_decoupling"]
    },
    {
        "exp_name": "Exp_v11_wo_SSL_s3407",
        "flags": ["--no_decoupling"],
        "override_ssl": "0.0" 
    }
]

# ================= 执行逻辑 =================
def main():
    print(f"🚀 开始在 GPU {GPU_ID} 上执行 Seed 3407 的消融实验 (使用 main22.py)...")
    print("="*60)

    for i, exp in enumerate(experiments):
        exp_name = exp["exp_name"]
        print(f"\n▶️ [{i+1}/{len(experiments)}] 正在运行: {exp_name}")
        
        train_cmd = [PYTHON_EXEC, TARGET_SCRIPT, "--exp_name", exp_name]
        current_params = BASE_PARAMS.copy()
        
        if "override_ssl" in exp:
            for j, p in enumerate(current_params):
                if p == "--lambda_ssl":
                    current_params[j+1] = exp["override_ssl"]
                    
        train_cmd.extend(current_params)
        if "flags" in exp: train_cmd.extend(exp["flags"])

        print(f"   [Train] 执行命令: {' '.join(train_cmd)}")
        
        start_time = time.time()
        try:
            subprocess.run(train_cmd, check=True)
            duration = (time.time() - start_time) / 60
            print(f"✅ 实验 {exp_name} 训练完成！耗时: {duration:.2f} 分钟")
            
            print(f"📊 正在自动测试 {exp_name}...")
            run_dir = os.path.join("results", exp_name)
            
            # 测试脚本：因为你没改 test.py 的 seed，默认会用 777 测，但没关系，测试集是一样的
            test_cmd = [PYTHON_EXEC, TEST_SCRIPT, "--run_dir", run_dir, "--gpu", GPU_ID]
            if "flags" in exp: test_cmd.extend(exp["flags"])
                
            print(f"   [Test] 执行命令: {' '.join(test_cmd)}")
            subprocess.run(test_cmd, check=True)
            print(f"✅ 实验 {exp_name} 测试完成！")
            
        except subprocess.CalledProcessError as e:
            print(f"❌ 实验 {exp_name} 失败！退出代码: {e.returncode}")

if __name__ == "__main__":
    main()