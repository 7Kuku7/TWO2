import subprocess
import time
import sys
import os

# ================= 配置区域 =================
PYTHON_EXEC = sys.executable 
TARGET_SCRIPT = "main2.py"
TEST_SCRIPT = "test_ablation111.py"  # 确保这个测试脚本是你修复过的那版
GPU_ID = "0" 

# 统一使用我们在 v8 中找到的“黄金超参数”
# 为了防爆显存，继续保持 BATCH_SIZE=2
BASE_PARAMS = [
    "--gpu", GPU_ID,
    "--epochs", "200",
    "--batch_size", "2",
    "--lambda_rank", "0.2",   
    "--lambda_ssl", "0.1",    
    "--lambda_sub", "0.05"
]

# ================= 实验计划 (Exp_v10 终极架构消融) =================
experiments = [
    # 1. Model A (Baseline): 无Fusion，无MultiTask
    {
        "exp_name": "Exp_v11_Model_A_Baseline",
        "flags": ["--no_fusion", "--no_multitask", "--no_decoupling"] 
    },
    
    # 2. Model B (MT Only): 有MultiTask，无Fusion
    {
        "exp_name": "Exp_v11_Model_B_MT_Only",
        "flags": ["--no_fusion", "--no_decoupling"]
    },

    # 3. Model C (Fusion Only): 有Fusion，无MultiTask
    {
        "exp_name": "Exp_v11_Model_C_Fusion_Only",
        "flags": ["--no_multitask", "--no_decoupling"]
    },

    # 4. Ours (Proposed): 有Fusion，有MultiTask (你最终的完全体模型)
    {
        "exp_name": "Exp_v11_Ours_Proposed",
        "flags": ["--no_decoupling"]
    },
    
    # 5. 附加验证：证明你设计的 SSL 自监督模块是有效的 (关闭 SSL)
    {
        "exp_name": "Exp_v11_wo_SSL",
        "flags": ["--no_decoupling"],
        "override_ssl": "0.0"  # 专门把 ssl 权重设为 0
    }
]

# ================= 执行逻辑 =================
def main():
    print(f"🚀 开始执行 Exp_v11 终极架构消融实验 (包含验证 SSL)...")
    print("="*60)

    for i, exp in enumerate(experiments):
        exp_name = exp["exp_name"]
        print(f"\n▶️ [{i+1}/{len(experiments)}] 正在运行: {exp_name}")
        
        # 1. 构造训练命令
        train_cmd = [PYTHON_EXEC, TARGET_SCRIPT, "--exp_name", exp_name]
        
        # 拷贝基础参数
        current_params = BASE_PARAMS.copy()
        
        # 如果需要覆盖 SSL 参数 (针对第5个实验)
        if "override_ssl" in exp:
            for j, p in enumerate(current_params):
                if p == "--lambda_ssl":
                    current_params[j+1] = exp["override_ssl"]
                    
        train_cmd.extend(current_params)
        
        if "flags" in exp:
            train_cmd.extend(exp["flags"])

        print(f"   [Train] 执行命令: {' '.join(train_cmd)}")
        
        start_time = time.time()
        try:
            # 执行训练
            subprocess.run(train_cmd, check=True)
            duration = (time.time() - start_time) / 60
            print(f"✅ 实验 {exp_name} 训练完成！耗时: {duration:.2f} 分钟")
            
            # 2. 自动执行测试
            print(f"📊 正在自动测试 {exp_name}...")
            # 构造测试目录
            run_dir = os.path.join("results", exp_name)
            test_cmd = [PYTHON_EXEC, TEST_SCRIPT, "--run_dir", run_dir]
            
            # 测试时必须带上同样的模型结构开关
            if "flags" in exp:
                test_cmd.extend(exp["flags"])
                
            print(f"   [Test] 执行命令: {' '.join(test_cmd)}")
            subprocess.run(test_cmd, check=True)
            print(f"✅ 实验 {exp_name} 测试完成！测试结果已保存。")
            
        except subprocess.CalledProcessError as e:
            print(f"❌ 实验 {exp_name} 失败！退出代码: {e.returncode}")

    print("\n🎉 终极消融实验计划执行完毕！可以直接提取 test_results.json 填入论文表格了！")

if __name__ == "__main__":
    main()