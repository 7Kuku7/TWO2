import subprocess
import time
import sys

# ================= 配置区域 =================
# Python解释器路径 (确保使用当前虚拟环境的python)
PYTHON_EXEC = sys.executable 

# 指定要运行的核心脚本 (即你的消融实验版 main)
TARGET_SCRIPT = "main2.py"

# 指定 GPU ID
GPU_ID = "0" 

# ================= 实验计划 (Exp_v8) =================
# 基于 Seed 3407 的最佳结果 (Exp_v7) 进行微调
# 基准配置 (Baseline): 
#   - MI=0.0 (即 --no_decoupling)
#   - Fusion=False (即 --no_fusion)
#   - Rank=0.1, SSL=0.2, Sub=0.05

experiments = [
    # --- 1. Baseline 复现 (确保环境正常，且作为对比基准) ---
    {
        "exp_name": "Exp_v8_Baseline",
        "params": {
            "--lambda_rank": 0.1,
            "--lambda_ssl": 0.2,
            "--lambda_sub": 0.05,
        },
        # 根据 v7 结论：不使用 Fusion，不使用 Decoupling (MI)
        "flags": ["--no_fusion", "--no_decoupling"] 
    },

    # --- 2. Rank Loss 权重探究 ---
    {
        "exp_name": "Exp_v8_Rank0.05",
        "params": {
            "--lambda_rank": 0.05, # 降低
            "--lambda_ssl": 0.2,
            "--lambda_sub": 0.05,
        },
        "flags": ["--no_fusion", "--no_decoupling"]
    },
    {
        "exp_name": "Exp_v8_Rank0.2",
        "params": {
            "--lambda_rank": 0.2,  # 升高
            "--lambda_ssl": 0.2,
            "--lambda_sub": 0.05,
        },
        "flags": ["--no_fusion", "--no_decoupling"]
    },

    # --- 3. SSL 自监督权重探究 ---
    {
        "exp_name": "Exp_v8_SSL0.1",
        "params": {
            "--lambda_rank": 0.1,
            "--lambda_ssl": 0.1,   # 降低
            "--lambda_sub": 0.05,
        },
        "flags": ["--no_fusion", "--no_decoupling"]
    },
    {
        "exp_name": "Exp_v8_SSL0.5",
        "params": {
            "--lambda_rank": 0.1,
            "--lambda_ssl": 0.5,   # 升高
            "--lambda_sub": 0.05,
        },
        "flags": ["--no_fusion", "--no_decoupling"]
    },

    # --- 4. Sub-score 子任务权重探究 ---
    {
        "exp_name": "Exp_v8_Sub0.1",
        "params": {
            "--lambda_rank": 0.1,
            "--lambda_ssl": 0.2,
            "--lambda_sub": 0.1,   # 升高
        },
        "flags": ["--no_fusion", "--no_decoupling"]
    },
]

# ================= 执行逻辑 =================
def main():
    print(f"🚀 开始执行 Exp_v8 网格搜索，共 {len(experiments)} 个实验...")
    print(f"📌 核心脚本: {TARGET_SCRIPT}")
    print(f"📌 GPU: {GPU_ID}")
    print("="*60)

    for i, exp in enumerate(experiments):
        exp_name = exp["exp_name"]
        print(f"\n▶️ [{i+1}/{len(experiments)}] 正在运行: {exp_name}")
        
        # 1. 构造基础命令
        cmd = [
            PYTHON_EXEC, TARGET_SCRIPT,
            "--exp_name", exp_name,
            "--gpu", GPU_ID,
            "--epochs", "200" # 统一 Epoch 数
        ]
        
        # 2. 添加数值参数 (Loss Weights)
        for key, value in exp["params"].items():
            cmd.append(key)
            cmd.append(str(value))
            
        # 3. 添加布尔开关 (Flags)
        # 注意：main2.py 里如果加了 --no_fusion，则 use_fusion=False
        if "flags" in exp:
            cmd.extend(exp["flags"])

        # 打印完整命令供检查
        print(f"   执行命令: {' '.join(cmd)}")
        
        # 4. 调用子进程运行 main2.py
        start_time = time.time()
        try:
            # check=True 表示如果有报错，会抛出异常
            subprocess.run(cmd, check=True)
            duration = (time.time() - start_time) / 60
            print(f"✅ 实验 {exp_name} 完成！耗时: {duration:.2f} 分钟")
            
        except subprocess.CalledProcessError as e:
            print(f"❌ 实验 {exp_name} 失败！退出代码: {e.returncode}")
            # 如果想遇到错误继续跑下一个，请注释掉下面这行 break
            # break 
        except Exception as e:
            print(f"❌ 发生未知错误: {e}")
            break

    print("\n🎉 所有实验计划执行完毕！")

if __name__ == "__main__":
    main()