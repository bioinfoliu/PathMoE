import json
import numpy as np
import pandas as pd

def json_to_csv():
    # 读取昨天跑完已经保存好的 JSON 数据
    json_file = "results/baseline_top2k_summary.json"
    try:
        with open(json_file, "r") as f:
            results = json.load(f)
    except FileNotFoundError:
        print(f"❌ 找不到文件: {json_file}")
        return

    models = ["Logistic Regression", "Random Forest", "XGBoost", "MLP"]
    table_data = []

    # 重新计算均值和标准差
    for cancer, m_data in results.items():
        row = {"Cancer": cancer}
        for m in models:
            # 过滤掉无效值
            aucs = [a for a in m_data[m] if a is not None and not np.isnan(a)]
            row[m] = f"{np.mean(aucs):.3f} ± {np.std(aucs):.3f}" if aucs else "—"
        table_data.append(row)

    # 导出为 CSV
    df = pd.DataFrame(table_data)
    out_csv = "results/baseline_top2k_summary_formatted.csv"
    df.to_csv(out_csv, index=False)
    print(f"✅ 太棒了！格式化的 CSV 表格已成功保存至: {out_csv}")

if __name__ == "__main__":
    json_to_csv()