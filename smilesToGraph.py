from rdkit import Chem
from rdkit.Chem import AllChem
import torch
from torch_geometric.data import Data
import hashlib




# 将SMILES转换为图数据的函数
def smiles_to_graph_no_3D_structure(smiles, target_chembl_id, molecule_chembl_id, ligand_efficiency, value):
    """
    将SMILES字符串转换为图数据对象。

    参数:
    - smiles (str): 分子的SMILES表示
    - target_chembl_id (str): 靶点的ChEMBL ID
    - molecule_chembl_id (str): 分子的ChEMBL ID
    - ligand_efficiency (dict): 配体效率信息
    - value (float): 目标值

    返回:
    - graph (Data): 图数据对象
    """

    # 解析SMILES
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError("Invalid SMILES")  # 如果解析失败，抛出异常

    # 添加隐式氢原子
    mol = Chem.AddHs(mol)

    # 获取节点（原子）特征
    node_features = []
    for idx, atom in enumerate(mol.GetAtoms()):
        node_features.append([atom.GetAtomicNum()])
    node_features = torch.tensor(node_features, dtype=torch.float)  # 转换为PyTorch张量

    # 获取边（键）信息
    edge_index = []
    edge_features = []
    for bond in mol.GetBonds():
        # 获取连接的原子索引
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        # 添加边（无向图）
        edge_index.append([i, j])
        edge_index.append([j, i])
        # 使用键的类型作为边特征
        edge_features.append([bond.GetBondTypeAsDouble()])
        edge_features.append([bond.GetBondTypeAsDouble()])

    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()  # 转换为PyTorch张量并转置
    edge_features = torch.tensor(edge_features, dtype=torch.float)  # 转换为PyTorch张量

    # 将字符串ID转换为数值特征
    def hash_id(id_str):
        return int(hashlib.md5(id_str.encode()).hexdigest(), 16) % 1000000

    molecule_id_feature = hash_id(molecule_chembl_id)
    target_id_feature = hash_id(target_chembl_id)
    # print(ligand_efficiency)

    # 提取配体效率信息
    # 添加类型检查
    # if isinstance(ligand_efficiency, dict):
    #     ligand_efficiency_values = [float(ligand_efficiency.get(key, 0.0)) for key in ['bei', 'le', 'lle', 'sei']]
    # else:
    #     print("Error: ligand_efficiency is not a dictionary.")
    #     ligand_efficiency_values = [0.0] * 4  # 设置默认值

    # ligand_efficiency_features = torch.tensor(ligand_efficiency_values, dtype=torch.float)

    # 将数值特征与节点特征拼接
    additional_features = torch.tensor([molecule_id_feature, target_id_feature], dtype=torch.float)
    additional_features = additional_features.repeat(node_features.size(0), 1)
    node_features = torch.cat([node_features, additional_features], dim=1)

    # 创建图数据对象
    graph = Data(
        x=node_features,  # 节点特征
        edge_index=edge_index,  # 边索引
        edge_attr=edge_features,  # 边特征
        y=torch.tensor([value], dtype=torch.float),  # 目标值
        ligand_efficiency=ligand_efficiency  # 配体效率特征
    )
    print(graph)  # 打印图数据对象
    return graph


smiles = "CC"
value = 4.7
target_chembl_id = "CHEMBL341591"
molecule_chembl_id = "MOL12345"

# graph = smiles_to_graph_no_3D_structure(smiles, target_chembl_id, molecule_chembl_id,'a', value)
# if graph is not None:
#     print(graph.x)  # 打印节点特征
#     print(graph.y)  # 打印目标值
#     print(graph.edge_index)
#     print(graph.edge_attr)
#     print(graph.morgan_fp)  # 打印Morgan指纹特征
