"""
化合物活性数据筛选系统（单靶点）
Created on Fri Mar 21 09:00:00 2025
"""
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors
import matplotlib.pyplot as plt
from TrainerGraph import getPath
import time
import os

# ======================
# 配置参数（按需修改）
# ======================
INPUT_FILE = os.path.join(getPath(), "result", "predictions_20250321_134511.csv")
OUTPUT_FILE = getPath() + "/result/filtered_compounds.parquet"
ACTIVITY_THRESHOLD = 0.1  # 归一化IC50筛选阈值
TOP_COMPOUNDS = 20  # 全局保留前N个活性最高化合物

# ======================
# 数据预处理函数
# ======================
def clean_data(df):
    """数据清洗与格式转换"""
    print(f"🛠️ 初始数据量：{len(df)} 条")

    # 删除关键字段缺失值
    df = df.dropna(subset=['smiles', 'predicted_value'])
    print(f"  缺失值过滤后：{len(df)} 条")

    # 过滤异常值
    df = df[(df['predicted_value'] >= 0) & (df['predicted_value'] <= 1)]

    # 移除靶点信息
    if 'target_chembl_id' in df.columns:
        df = df.drop(columns=['target_chembl_id'])

    # 计算分子量
    df['mol'] = df['smiles'].apply(lambda x: Chem.MolFromSmiles(str(x)))
    df['MolWt'] = df['mol'].apply(Descriptors.MolWt)

    # 过滤大分子
    df = df[df['MolWt'] <= 900]
    print(f"  分子量过滤后：{len(df)} 条")

    return df.drop_duplicates('smiles')

# ======================
# 核心筛选逻辑
# ======================
def main():
    start_time = time.time()

    # 初始化输出目录
    output_dir = os.path.dirname(OUTPUT_FILE)
    os.makedirs(output_dir, exist_ok=True)

    print("🚀 启动化合物筛选系统（单次处理模式）...")
    print(f"输入文件：{INPUT_FILE}")
    print(f"输出格式：Parquet ({OUTPUT_FILE}) + CSV ({OUTPUT_FILE.replace('.parquet', '.csv')})")

    # 读取全部数据
    full_df = pd.read_csv(INPUT_FILE)

    # 执行数据清洗
    cleaned_df = clean_data(full_df)

    # 执行筛选逻辑
    filtered_df = cleaned_df[cleaned_df['predicted_value'] <= ACTIVITY_THRESHOLD]
    final_df = filtered_df.nsmallest(TOP_COMPOUNDS, 'predicted_value')

    # ======================
    # 结果保存与可视化
    # ======================
    print(f"\n🎯 最终筛选结果：{len(final_df)} 条化合物")
    final_df = final_df.drop(columns=['mol'])  # 保存数据前移除mol列
    # 保存数据
    final_df.to_parquet(OUTPUT_FILE)
    final_df.to_csv(getPath()+"/result/filter_result.csv", index=False, encoding='utf-8-sig')


    print(f"Parquet文件：{os.path.abspath(OUTPUT_FILE)}")


if __name__ == "__main__":
    main()
