from chembl_webresource_client.new_client import new_client
import pandas as pd
from utils.util import getPath
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import time


def pIC50(input):
    pIC50 = []
    for i in input:
        molar = i * (10 ** -9)  # Converts nM to M
        pIC50.append(-np.log10(molar))
    return pIC50


def download_data(selected_target):
    activity = new_client.activity
    # res = activity.filter(target_chembl_id=selected_target).filter(standard_type="IC50")
    res = activity.filter(target_chembl_id=selected_target)
    df = pd.DataFrame.from_dict(res)
    if df.empty:
        print("没有找到任何数据")
        return
    print(df)
    # 如果任何化合物在**standard_value**列中缺少值，则删除它
    # df = df[df.standard_value.notna()]
    # df = df[df.ligand_efficiency.notna()]
    # df = df[df.pchembl_value.notna()]

    # columns_to_keep = ['molecule_chembl_id', 'canonical_smiles', 'canonical_smiles', 'ligand_efficiency',
    #                    'target_chembl_id', 'assay_type', 'standard_value', 'pchembl_value', 'standard_units']
    # # df2['pchembl_value'] = pchembl_values
    # df = df[columns_to_keep]
    # # 初始化MinMaxScaler
    # scaler = MinMaxScaler()
    # # 对standard_value列进行归一化
    # df['standard_value_normalized'] = scaler.fit_transform(df[['standard_value']])
    # 对standard_value列进行pIC50归一化
    # IC50 = pIC50(df['standard_value'])
    # df['pIC50'] = IC50
    return df


def get_target_info(target_chembl_name):
    target = new_client.target
    #  在ChEMBL数据库中搜索对应的靶点信息。搜索结果会返回包含所查询靶点详细信息的对象列表，例如靶点的名称、描述、相关的化合物等
    target_query = target.search(target_chembl_name)
    # target_query = target.filter(therapeutic_area=target_chembl_name)
    targets = pd.DataFrame.from_dict(target_query)
    print(targets)
    df_list = []
    for index, row in targets.iterrows():
        if row['cross_references']:
            print(index)
            selected_target = targets.target_chembl_id[index]
            print(selected_target)
            time.sleep(30)  # 等待30秒再去请求
            df_list.append(download_data(selected_target))

    # 合并所有DataFrame
    if df_list:
        all_df = pd.concat(df_list, ignore_index=True)
        file_name = f'data_{target_chembl_name}.csv'
        all_df.to_csv(getPath() + f'/data/rawData/{file_name}', index=False)
        return all_df
    else:
        print("没有找到任何数据")
        return pd.DataFrame()


# 示例调用 aromatase coronavirus
# 定义一个包含多种疾病相关蛋白名称的列表
list_diseases = [
    # "cyclooxygenase",  # 环氧合酶
    # "estrogen receptor",  # 雌激素受体
    "kinase",  # 激酶
    "gaba receptor",  # GABA受体
    "dopamine receptor",  # 多巴胺受体
    "acetylcholinesterase",  # 乙酰胆碱酯酶
    "histamine receptor",  # 组胺受体
    "insulin receptor",  # 胰岛素受体
    "nuclear receptor",  # 核受体
    "serotonin receptor",  # 5-羟色胺受体
    "androgen receptor",  # 雄激素受体
    "thyroid hormone receptor",  # 甲状腺激素受体
    "phosphodiesterase",  # 磷酸二酯酶
    "angiotensin receptor",  # 血管紧张素受体
    "integrase",  # 整合酶
    "protease",  # 蛋白酶
    "reverse transcriptase",  # 逆转录酶
    "carbonic anhydrase",  # 碳酸酐酶
    "cholesterol esterase",  # 胆固醇酯酶
    "glucocorticoid receptor",  # 糖皮质激素受体
    "glutamate receptor",  # 谷氨酸受体
    "heat shock protein",  # 热休克蛋白
    "opioid receptor",  # 阿片受体
    "peroxisome proliferator-activated receptor",  # 过氧化物酶体增殖物激活受体
    "progesterone receptor",  # 孕激素受体
    "retinoic acid receptor",  # 视黄酸受体
    "sigma receptor",  # 西格玛受体
    "somatostatin receptor",  # 生长抑素受体
    "hiv",  # 艾滋病病毒
    "diabetes",  # 糖尿病
    "aromatase",  # 香豆素酶
    "coronavirus"  # 冠状病毒
]

# 过氧化物酶体增殖物激活受体γ（Peroxisome proliferator-activated receptor gamma，PPAR-γ）。
# PPAR-γ在糖尿病管理中起着关键作用，因为它可以增强胰岛素敏感性并调节葡萄糖代谢。尽管PPAR-γ激动剂如噻唑烷二酮类药物在解决糖尿病并发症方面有效，
# 但它们也与一些可能引发健康问题的副作用相关

combined_df = get_target_info('PPAR-γ')
print(combined_df.head())

# for disease in list_diseases:
#     combined_df = get_target_info('coronavirus')
#     print(combined_df.head())
