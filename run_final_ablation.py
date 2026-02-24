import subprocess
import time
import sys

# ================= 配置区域 =================
PYTHON_EXEC = sys.executable 
TARGET_SCRIPT = "main2.py"
GPU_ID = "0" 

# 统一使用我们在 v8 中找到的“黄金超参数”
# 为了防爆显存，继续保持 BATCH_SIZE=2
BEST_PARAMS = [
    "--gpu", GPU_ID,
    "--epochs", "200",
    "--batch_size", "2",
    "--lambda_rank", "0.2",   # 结合了 v8 最优结果
    "--lambda_ssl", "0.1",    # 结合了 v8 最优结果
    "--lambda_sub", "0.05"
]

# ================= 实验计划 (Exp_v10 最终架构消融) =================
experiments = [
    # 1. 你提出的全量完整模型 (Our Proposed Full Model)
    # 期望结果：最高分
    {
        "exp_name": "Exp_v10_Ours_Full",
        "flags": [] # 不加任何 no_ 标志，默认全部开启 (Fusion=True, Decoupling=True, MultiTask=True)
    },

    # 2. 去掉自适应特征融合 (用 Concat 替代)
    {
        "exp_name": "Exp_v10_wo_Fusion",
        "flags": ["--no_fusion"] 
    },

    # 3. 去掉 MI 特征解耦约束
    {
        "exp_name": "Exp_v10_wo_Decoupling",
        "flags": ["--no_decoupling"]
    },

    # 4. 去掉多任务辅助分支 (Sub-score)
    {
        "exp_name": "Exp_v10_wo_MultiTask",
        "flags": ["--no_multitask"]
    },
    
    # 注意：单分支消融（仅Content或仅Distortion）可能需要修改模型代码，
    # 建议先跑完上面这四个核心模块的消融。
]

# ================= 执行逻辑 =================
def main():
    print(f"🚀 开始执行 Exp_v10 最终架构消融实验...")
    print("="*60)

    for i, exp in enumerate(experiments):
        exp_name = exp["exp_name"]
        print(f"\n▶️ [{i+1}/{len(experiments)}] 正在运行: {exp_name}")
        
        cmd = [PYTHON_EXEC, TARGET_SCRIPT, "--exp_name", exp_name] + BEST_PARAMS
        if "flags" in exp:
            cmd.extend(exp["flags"])

        print(f"   执行命令: {' '.join(cmd)}")
        
        start_time = time.time()
        try:
            subprocess.run(cmd, check=True)
            duration = (time.time() - start_time) / 60
            print(f"✅ 实验 {exp_name} 完成！耗时: {duration:.2f} 分钟")
        except subprocess.CalledProcessError as e:
            print(f"❌ 实验 {exp_name} 失败！退出代码: {e.returncode}")

    print("\n🎉 最终消融实验计划执行完毕！如果 Ours_Full 跑得比其他的都高，你的论文核心实验就做完了！")

if __name__ == "__main__":
    main()