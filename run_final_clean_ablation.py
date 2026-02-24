import subprocess
import sys
import time

PYTHON_EXEC = sys.executable 
TARGET_SCRIPT = "main2.py"
GPU_ID = "0" 

# 统一使用最强超参数
BEST_PARAMS = [
    "--gpu", GPU_ID,
    "--epochs", "200",
    "--batch_size", "2",
    "--lambda_rank", "0.2",   
    "--lambda_ssl", "0.1",    
    "--lambda_sub", "0.05"
]

experiments = [
    # 对应表格里的 Model A (最素的基准：无Fusion，无MultiTask，无MI)
    {
        "exp_name": "Exp_Final_Model_A_Base",
        "flags": ["--no_fusion", "--no_multitask", "--no_decoupling"] 
    },
    # 对应表格里的 Model C (只有Fusion：无MultiTask，无MI)
    {
        "exp_name": "Exp_Final_Model_C_FusionOnly",
        "flags": ["--no_multitask", "--no_decoupling"]
    }
]

def main():
    print("🚀 开始补全终极消融实验的最后两块拼图...")
    for exp in experiments:
        exp_name = exp["exp_name"]
        print(f"\n▶️ 正在运行: {exp_name}")
        cmd = [PYTHON_EXEC, TARGET_SCRIPT, "--exp_name", exp_name] + BEST_PARAMS + exp["flags"]
        
        start_time = time.time()
        try:
            subprocess.run(cmd, check=True)
            print(f"✅ 实验 {exp_name} 训练完成！耗时: {(time.time() - start_time) / 60:.2f} 分钟")
            
            # 训练完自动调用测试脚本
            print(f"📊 正在测试 {exp_name}...")
            test_cmd = [PYTHON_EXEC, "test_ablation111.py", "--run_dir", f"results/{exp_name}"] + exp["flags"]
            subprocess.run(test_cmd, check=True)
            
        except subprocess.CalledProcessError as e:
            print(f"❌ 实验 {exp_name} 失败！")

if __name__ == "__main__":

    main()