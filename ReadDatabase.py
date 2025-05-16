import sqlite3
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski

from utils.util import getPath

def lipinski(smile):
    mol = Chem.MolFromSmiles(smile)
    if mol is None:
        return False
    return all([
        Descriptors.MolWt(mol) < 500,
        Descriptors.MolLogP(mol) < 5,
        Lipinski.NumHDonors(mol) < 5,
        Lipinski.NumHAcceptors(mol) < 10
    ])


def get_bioactivity_data(limit=10, offset=0):
    """
    获取包含关键生物活性数据的查询结果（最终版）
    参数:
        limit: 每次查询返回的最大记录数
        offset: 分页偏移量
    返回:
        DataFrame: 包含目标字段的结构化数据
    """
    conn = None
    try:
        conn = sqlite3.connect(getPath() + '/chembl_34_sqlite/chembl_34.db')

        # 包含standard_value的完整查询
        query = """
        SELECT 
            cs.canonical_smiles AS smiles,
            md.chembl_id AS molecule_chembl_id,
            td.chembl_id AS target_chembl_id,
            act.standard_value,  -- 新增标准值字段
            le.bei, 
            le.le, 
            le.lle, 
            le.sei
        FROM activities act
        JOIN assays a ON act.assay_id = a.assay_id
        JOIN molecule_dictionary md ON act.molregno = md.molregno
        JOIN compound_structures cs ON md.molregno = cs.molregno
        JOIN target_dictionary td ON a.tid = td.tid
        JOIN ligand_eff le ON act.activity_id = le.activity_id
        WHERE act.standard_type = 'IC50'
          AND act.standard_value IS NOT NULL
        LIMIT ? OFFSET ?;
        """

        # 执行查询
        df = pd.read_sql_query(query, conn, params=(limit, offset))

        # 更新列名顺序
        columns = ['smiles', 'molecule_chembl_id', 'target_chembl_id',
                  'standard_value', 'bei', 'le', 'lle', 'sei']  # 调整列顺序
        df = df[columns]

        if df.empty:
            print("警告：查询到0条记录")

        return df

    except sqlite3.Error as e:
        print(f"数据库操作异常: {str(e)}")
        return pd.DataFrame()
    finally:
        if conn:
            conn.close()

def get_total_records():
    """获取符合条件总记录数（同步更新字段）"""
    conn = None
    try:
        conn = sqlite3.connect(getPath() + '/chembl_34_sqlite/chembl_34.db')
        query = """
        SELECT COUNT(*) 
        FROM activities act
        JOIN assays a ON act.assay_id = a.assay_id
        JOIN molecule_dictionary md ON act.molregno = md.molregno
        JOIN compound_structures cs ON md.molregno = cs.molregno
        JOIN target_dictionary td ON a.tid = td.tid
        JOIN ligand_eff le ON act.activity_id = le.activity_id
        WHERE act.standard_type = 'IC50'
          AND act.standard_value IS NOT NULL
        """
        return pd.read_sql_query(query, conn).iloc[0,0]
    except sqlite3.Error as e:
        print(f"计数查询失败: {str(e)}")
        return -1
    finally:
        if conn:
            conn.close()
def get_bioactivity_without_standard(limit=10, offset=0):
    """
    获取不含standard_value字段的生物活性数据
    参数:
        limit: 每次查询返回的最大记录数
        offset: 分页偏移量
    返回:
        DataFrame: 包含基础字段的结构化数据
    """
    conn = None
    try:
        conn = sqlite3.connect(getPath() + '/chembl_34_sqlite/chembl_34.db')

        # 去除了standard_value的查询
        query = """
        SELECT 
            cs.canonical_smiles AS smiles,
            md.chembl_id AS molecule_chembl_id,
            td.chembl_id AS target_chembl_id,
            le.bei, 
            le.le, 
            le.lle, 
            le.sei
        FROM activities act
        JOIN assays a ON act.assay_id = a.assay_id
        JOIN molecule_dictionary md ON act.molregno = md.molregno
        JOIN compound_structures cs ON md.molregno = cs.molregno
        JOIN target_dictionary td ON a.tid = td.tid
        JOIN ligand_eff le ON act.activity_id = le.activity_id
        WHERE act.standard_type = 'IC50'
        LIMIT ? OFFSET ?;
        """

        df = pd.read_sql_query(query, conn, params=(limit, offset))
        print("数据长度是："+pd.read_sql_query(query, conn).iloc[0, 0])
        return df[['smiles', 'molecule_chembl_id', 'target_chembl_id',
                 'bei', 'le', 'lle', 'sei']]  # 显式过滤列

    except sqlite3.Error as e:
        print(f"精简版查询异常: {str(e)}")
        return pd.DataFrame()
    finally:
        if conn:
            conn.close()

if __name__ == "__main__":
    try:
        total = get_total_records()
        print(f"总有效记录数: {total if total != -1 else '查询失败'}")

        df = get_bioactivity_data(limit=916747)
        # df = get_bioactivity_without_standard(limit=999999999,offset=0)  # 精简版
        df = df[df['smiles'].apply(lipinski)]  # 新增过滤行

        print("\n数据样例：")
        print(df.head(3) if not df.empty else "无有效数据")

        if not df.empty:
            print("\n字段说明：")
            print(df.dtypes)  # 新增字段验证

            if not df.empty:
                import numpy as np

                # 智能分割（自动处理余数分布）
                split_dfs = np.array_split(df, 10)

                # 分文件保存
                for i, split_df in enumerate(split_dfs, 1):
                    filename = f'bioactivity_data_part{i}.csv'
                    split_df.to_csv(filename, index=False)
                    print(f'已保存: {filename} ({len(split_df)}条记录)')

            print("\n保存成功")

    except Exception as e:
        print(f"运行异常: {str(e)}")
