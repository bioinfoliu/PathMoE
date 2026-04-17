import os
import pandas as pd
import numpy as np

# 如果没有安装 GEOparse，需要先运行: pip install GEOparse
try:
    import GEOparse
except ImportError:
    print("GEOparse is not installed. Please install it using: pip install GEOparse")
    exit()

def download_and_process_geo(gse_id="GSE25066", dest_dir="./geo_data"):
    # 1. 创建保存目录
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)

    # 2. 下载并解析 GEO 数据
    print(f"正在下载并解析 {gse_id} 数据，这可能需要几分钟时间...")
    gse = GEOparse.get_GEO(geo=gse_id, destdir=dest_dir)

    # 3. 提取临床元数据 (Phenotype Data)
    print("正在提取临床标签数据...")
    pheno_data = gse.phenotype_data
    
    # 自动寻找包含亚型信息的列
    subtype_col = None
    for col in pheno_data.columns:
        if 'pam50' in col.lower() or 'subtype' in col.lower():
            subtype_col = col
            break

    if subtype_col:
        labels = pheno_data[[subtype_col]].copy()
        labels.rename(columns={subtype_col: 'PAM50_Subtype'}, inplace=True)
        # 清理字符串内容 (例如把 "pam50 subtype: Luminal A" 变成 "Luminal A")
        labels['PAM50_Subtype'] = labels['PAM50_Subtype'].apply(
            lambda x: str(x).split(':')[-1].strip() if pd.notnull(x) else np.nan
        )
    else:
        print("警告: 未能自动识别到明确的 PAM50 标签列，将返回所有临床特征，请手动核对。")
        labels = pheno_data

    # 4. 提取表达矩阵 (Expression Matrix)
    print("正在提取表达矩阵...")
    gsm_names = list(gse.gsms.keys())
    expr_list = []

    for gsm_name in gsm_names:
        tmp_df = gse.gsms[gsm_name].table
        if not tmp_df.empty and 'ID_REF' in tmp_df.columns:
            tmp_df = tmp_df.set_index('ID_REF')
            # 提取具体的表达值列 (通常命名为 'VALUE')
            if 'VALUE' in tmp_df.columns:
                series = tmp_df['VALUE']
                series.name = gsm_name
                expr_list.append(series)

    # 将所有样本拼接成一个大型矩阵
    expr_matrix = pd.concat(expr_list, axis=1)

    # 5. 合并表达矩阵与标签
    print("正在合并表达矩阵与临床标签...")
    # 转置矩阵，使得行是样本 (Samples)，列是基因/探针 (Genes/Probes)
    expr_matrix_t = expr_matrix.T

    # 按照样本 ID (index) 进行内连接
    final_dataset = expr_matrix_t.merge(labels, left_index=True, right_index=True, how='inner')

    # 删除缺乏 PAM50 标签的缺失值行
    if 'PAM50_Subtype' in final_dataset.columns:
        final_dataset = final_dataset.dropna(subset=['PAM50_Subtype'])

    # 6. 保存为 CSV 文件
    output_file = os.path.join(dest_dir, f"{gse_id}_processed_dataset.csv")
    final_dataset.to_csv(output_file)
    
    print(f"数据处理完成！已成功保存至: {output_file}")
    print(f"最终数据集维度 (样本数, 特征数+标签): {final_dataset.shape}")
    
    return final_dataset

# 执行主函数
if __name__ == "__main__":
    dataset = download_and_process_geo(gse_id="GSE25066")
    print("\n数据集预览:")
    print(dataset.head())