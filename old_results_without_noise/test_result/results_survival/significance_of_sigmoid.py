import json
import os
import glob
import pandas as pd
from scipy.stats import wilcoxon

def run_statistical_test(sigmoid_dir, softmax_dir):
    # 获取所有的 Sigmoid (默认) 结果文件
    sigmoid_files = glob.glob(os.path.join(sigmoid_dir, "*_summary.json"))
    
    results_list = []

    print(f"{'Cancer':<10} | {'Sigmoid Mean':<12} | {'Softmax Mean':<12} | {'P-Value':<10} | {'Significant'}")
    print("-" * 75)

    for sig_file in sigmoid_files:
        filename = os.path.basename(sig_file)
        cancer = filename.split('_')[0]
        
        # 构建对应的 Softmax 文件路径
        soft_filename = f"{cancer}_softmax_summary.json"
        soft_file = os.path.join(softmax_dir, soft_filename)

        if not os.path.exists(soft_file):
            print(f"⚠️ 跳过 {cancer}: 找不到对应的 Softmax 文件 {soft_file}")
            continue

        # 读取 JSON 内容
        with open(sig_file, 'r') as f:
            sig_data = json.load(f)
        with open(soft_file, 'r') as f:
            soft_data = json.load(f)

        # 提取 20 个种子的 AUC 列表
        sig_aucs = sig_data.get("seed_test_aucs", [])
        soft_aucs = soft_data.get("seed_test_aucs", [])

        if len(sig_aucs) != len(soft_aucs) or len(sig_aucs) == 0:
            print(f"⚠️ 跳过 {cancer}: 种子数量不匹配或为空")
            continue

        # 执行 Wilcoxon Signed-Rank Test (配对检验)
        # alternative='two-sided' 检验两者是否有差异
        stat, p_val = wilcoxon(sig_aucs, soft_aucs)

        sig_mean = sig_data.get("mean_test_auc", 0)
        soft_mean = soft_data.get("mean_test_auc", 0)
        is_significant = "✅ YES" if p_val < 0.05 else "❌ NO"

        print(f"{cancer:<10} | {sig_mean:<12.4f} | {soft_mean:<12.4f} | {p_val:<10.4f} | {is_significant}")
        
        results_list.append({
            "Cancer": cancer,
            "Sigmoid_Mean": sig_mean,
            "Softmax_Mean": soft_mean,
            "P_Value": p_val,
            "Significant": p_val < 0.05
        })

    # 保存为 CSV 方便查看和绘图
    df = pd.DataFrame(results_list)
    df.to_csv("wilcoxon_test_results.csv", index=False)
    print(f"\n📊 统计结果已保存至: wilcoxon_test_results.csv")

if __name__ == "__main__":
    # 根据你的路径设置目录
    SIGMOID_PATH = "/data/zliu/Path_MoE/results/test_result/results_survival/"
    SOFTMAX_PATH = "/data/zliu/Path_MoE/results/test_result/results_survival/"
    
    run_statistical_test(SIGMOID_PATH, SOFTMAX_PATH)