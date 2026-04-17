## 这个代码是用来整理softmax 和sigmoid 两个实验的结果table

import json
import os
import numpy as np

def get_metrics(file_path):
    """从 JSON 中提取均值和标准差"""
    if not os.path.exists(file_path):
        return None, None
    with open(file_path, 'r') as f:
        data = json.load(f)
        # 这里的 key 对应你之前发给我的 JSON 结构
        mean = data.get('mean_test_auc')
        std = data.get('std_test_auc')
        return mean, std

def generate_latex_table():
    cancers = ["BLCA", "BRCA", "HNSC", "KIRC", "LGG", "LUAD", "LUSC", "PRAD", "STAD", "THCA"]
    
    softmax_results = []
    sigmoid_results = []

    print("% --- 自动生成的 LaTeX 表格内容 ---")
    print(r"\begin{tabular}{lcc}")
    print(r"\hline")
    print(r"\textbf{Cancer Type} & \textbf{Softmax} & \textbf{Sigmoid} \\")
    print(r"\hline")

    for cancer in cancers:
        # 定义文件路径
        s_max_path = f"/data/zliu/results_ablation/{cancer}_softmax_summary.json"
        s_ig_path = f"/data/zliu/results/{cancer}_summary.json"

        m1, s1 = get_metrics(s_max_path)
        m2, s2 = get_metrics(s_ig_path)

        if m1 is None or m2 is None:
            print(f"% 警告: {cancer} 数据缺失")
            continue

        softmax_results.append(m1)
        sigmoid_results.append(m2)

        # 格式化字符串，不带前导零，保留3位小数
        str1 = f"{m1:.3f}".lstrip('0') + f"({s1:.3f})".replace("(0.", "(. ") if s1 else "" # 简单处理前导零
        # 更严谨的去零方式
        fmt_m1 = f"{m1:.3f}".replace("0.", ".")
        fmt_s1 = f"{s1:.3f}".replace("0.", ".")
        fmt_m2 = f"{m2:.3f}".replace("0.", ".")
        fmt_s2 = f"{s2:.3f}".replace("0.", ".")

        # 比较大小并加粗
        if m1 > m2:
            row = f"{cancer} & \\textbf{{{fmt_m1}({fmt_s1})}} & {fmt_m2}({fmt_s2}) \\\\"
        else:
            row = f"{cancer} & {fmt_m1}({fmt_s1}) & \\textbf{{{fmt_m2}({fmt_s2})}} \\\\"
        
        print(row)

    # 计算最终平均值
    avg_m1 = np.mean(softmax_results)
    avg_m2 = np.mean(sigmoid_results)
    
    # 这里标准差可以用所有 seed 列表重新算，也可以简单取平均（通常建议用所有 seed 重新算更严谨）
    print(r"\hline")
    fmt_avg1 = f"{avg_m1:.3f}".replace("0.", ".")
    fmt_avg2 = f"{avg_m2:.3f}".replace("0.", ".")
    
    if avg_m1 > avg_m2:
        print(f"\\textbf{{Average}} & \\textbf{{{fmt_avg1}}} & {fmt_avg2} \\\\")
    else:
        print(f"\\textbf{{Average}} & {fmt_avg1} & \\textbf{{{fmt_avg2}}} \\\\")

    print(r"\hline")
    print(r"\end{tabular}")

if __name__ == "__main__":
    generate_latex_table()