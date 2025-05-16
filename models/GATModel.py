from torch_geometric.nn import GATConv, TopKPooling  # 导入图注意力卷积和TopK池化模块
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp  # 导入全局平均池化和全局最大池化函数
import torch.nn.functional as F  # 导入PyTorch神经网络功能模块
import torch.nn as nn  # 导入PyTorch神经网络模块
import torch  # 导入PyTorch库

# 多层卷积和池化：
# 使用了多层 GATConv 卷积层，每一层后面都跟着批量归一化（BatchNorm）和 TopK 池化（TopKPooling）。
# 这种多层卷积和池化的组合有助于逐步提取图的高层次特征，并减少计算复杂度。

# TopK 池化：
# 使用 TopKPooling 进行节点选择，保留最重要的节点。这有助于减少图的规模，同时保留关键信息，提高模型的效率和性能。

# 全局池化策略：
# 在每一层卷积和池化之后，使用全局最大池化（Global Max Pooling, GMP）和全局平均池化（Global Mean Pooling, GAP）来提取全局特征。
# 这两种池化方法的结合可以捕捉到不同类型的特征信息，增强模型的鲁棒性。

# 多阶段特征融合：
# 将不同层级的池化结果拼接在一起，形成一个综合的特征表示。这种多阶段特征融合的方法有助于捕获不同层次的图结构信息，提高模型的表达能力。

# 全连接层的设计：
# 使用两个全连接层进行最终的特征转换和回归输出。第一个全连接层将多阶段特征融合的结果映射到一个中间维度，第二个全连接层将中间特征映射到最终的输出。
# 这种设计有助于模型更好地拟合复杂的非线性关系。

# 定义 GNN 模型
input_dim = 3  # 输入特征维度
output_dim = 128  # 输出特征维度

# 定义 Self-Attention 模块
class SelfAttention(nn.Module):
    """
    自注意力机制模块，用于在图神经网络层之间捕获节点特征的全局依赖关系。
    参数:
    embed_dim (int): 嵌入维度，即输入特征的维度。
    """
    def __init__(self, embed_dim):
        super(SelfAttention, self).__init__()  # 调用父类构造函数
        self.query = nn.Linear(embed_dim, embed_dim)  # 查询矩阵
        self.key = nn.Linear(embed_dim, embed_dim)  # 键矩阵
        self.value = nn.Linear(embed_dim, embed_dim)  # 值矩阵
        self.softmax = nn.Softmax(dim=-1)  # Softmax激活函数

    def forward(self, x):
        """
        前向传播函数，计算自注意力机制的输出。
        参数:
        x (torch.Tensor): 输入特征张量。
        返回:
        torch.Tensor: 经过自注意力机制处理后的特征张量。
        """
        x = x.unsqueeze(1)  # 扩展维度 [batch_size, 1, embed_dim]
        Q = self.query(x)  # 计算查询矩阵
        K = self.key(x)  # 计算键矩阵
        V = self.value(x)  # 计算值矩阵
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / (x.size(-1) ** 0.5)  # 计算注意力得分
        attention_weights = self.softmax(attention_scores)  # 应用Softmax得到注意力权重
        output = torch.matmul(attention_weights, V)  # 计算加权和
        output = output.squeeze(1)  # 压缩维度 [batch_size, embed_dim]
        return output


class GNNModel(nn.Module):
    """
    GNN模型类，包含多层GAT卷积、批量归一化、TopK池化和全连接层。
    该模型旨在通过图注意力机制和全局池化策略，提取图结构数据的高层次特征，并通过全连接层进行回归预测。
    """
    def __init__(self):
        super(GNNModel, self).__init__()  # 调用父类构造函数
        self.conv1 = GATConv(input_dim, output_dim, heads=4)  # 第一层GAT卷积
        self.bn1 = nn.BatchNorm1d(output_dim * 4)  # 第一层批量归一化
        self.pool1 = TopKPooling(output_dim * 4, ratio=0.8)  # 第一层TopK池化
        self.attention1 = SelfAttention(output_dim * 8)  # 第一层自注意力机制

        self.conv2 = GATConv(output_dim * 4, output_dim, heads=4)  # 第二层GAT卷积
        self.bn2 = nn.BatchNorm1d(output_dim * 4)  # 第二层批量归一化
        self.pool2 = TopKPooling(output_dim * 4, ratio=0.8)  # 第二层TopK池化
        self.attention2 = SelfAttention(output_dim * 8)  # 第二层自注意力机制

        self.conv3 = GATConv(output_dim * 4, output_dim, heads=4)  # 第三层GAT卷积
        self.bn3 = nn.BatchNorm1d(output_dim * 4)  # 第三层批量归一化
        self.pool3 = TopKPooling(output_dim * 4, ratio=0.8)  # 第三层TopK池化
        self.attention3 = SelfAttention(output_dim * 8)  # 第三层自注意力机制

        self.fc1 = nn.Linear(output_dim * 8 * 3, output_dim)  # 第一个全连接层
        self.fc2 = nn.Linear(output_dim, 1)  # 第二个全连接层

        # 初始化模型参数
        self.reset_parameters()

    def reset_parameters(self):
        """
        重置模型参数，使用 Kaiming 正态初始化方法。
        """
        for m in self.modules():  # 遍历所有子模块
            if isinstance(m, (nn.Conv2d, nn.Linear)):  # 如果是卷积或线性层
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')  # 使用Kaiming正态初始化权重
                if m.bias is not None:  # 如果有偏置项
                    nn.init.zeros_(m.bias)  # 初始化偏置为零

    def forward(self, data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch  # 获取输入数据

        # 第一层
        x = F.relu(self.bn1(self.conv1(x, edge_index)))  # 第一层卷积 + 批量归一化 + ReLU激活
        x, edge_index, _, batch, _, _ = self.pool1(x, edge_index, None, batch)  # 第一层TopK池化
        x1 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)  # 拼接全局最大池化和全局平均池化的结果
        x1 = self.attention1(x1)  # 应用第一层自注意力机制

        # 第二层
        x = F.relu(self.bn2(self.conv2(x, edge_index)))  # 第二层卷积 + 批量归一化 + ReLU激活
        x, edge_index, _, batch, _, _ = self.pool2(x, edge_index, None, batch)  # 第二层TopK池化
        x2 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)  # 拼接全局最大池化和全局平均池化的结果
        x2 = self.attention2(x2)  # 应用第二层自注意力机制

        # 第三层
        x = F.relu(self.bn3(self.conv3(x, edge_index)))  # 第三层卷积 + 批量归一化 + ReLU激活
        x, edge_index, _, batch, _, _ = self.pool3(x, edge_index, None, batch)  # 第三层TopK池化
        x3 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)  # 拼接全局最大池化和全局平均池化的结果
        x3 = self.attention3(x3)  # 应用第三层自注意力机制

        # 拼接并输出
        x = torch.cat([x1, x2, x3], dim=1)  # 拼接三个层级的特征
        x = self.fc1(x)  # 第一个全连接层
        x = F.relu(x)  # ReLU激活
        x = self.fc2(x)  # 第二个全连接层
        return x  # 返回最终输出
