import os
import torch
import torch.optim as optim
from torch_geometric.loader import DataLoader
from tqdm import tqdm
import logging

from TrainerGraph import gnn_trainer
from models.GATModel import GNNModel
from dataProcess import YouPPARyChooseBinaryDataset
from utils.util import getPath
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt


# 提取特征向量
def extract_features(model, data_loader, device):
    model.eval()
    features = []
    with torch.no_grad():
        for data in data_loader:
            data = data.to(device)
            output = model(data)
            features.append(output.cpu().numpy())
    return np.concatenate(features, axis=0)


# 修改后的训练函数
def train_sklearn_model(model_type, train_features, train_labels, epochs=10):
    losses = []

    if model_type == "SVM":
        # 使用SGDRegressor替代SVR以支持增量学习
        model = SGDRegressor(max_iter=1, tol=None, learning_rate='constant', eta0=0.01)
        scaler = StandardScaler()
        train_features = scaler.fit_transform(train_features)

        for epoch in tqdm(range(epochs)):
            model.partial_fit(train_features, train_labels)
            predictions = model.predict(train_features)
            loss = np.mean((predictions - train_labels) ** 2)
            losses.append(loss)
            print(f'Epoch {epoch + 1}/{epochs} | SVM Loss: {loss:.4f}')  # 打印损失值

    elif model_type == "RF":
        # 使用warm_start逐步增加树的数量
        model = RandomForestRegressor(
            n_estimators=0,
            warm_start=True,
            random_state=42
        )

        for epoch in tqdm(range(epochs)):
            model.n_estimators += 10  # 每个epoch增加10棵树
            model.fit(train_features, train_labels)
            predictions = model.predict(train_features)
            loss = np.mean((predictions - train_labels) ** 2)
            losses.append(loss)
            print(f'Epoch {epoch + 1}/{epochs} | RF Loss: {loss:.4f}')  # 打印损失值

    elif model_type == "KNN":
        # KNN无法分epoch训练，仅记录单次训练结果
        model = KNeighborsRegressor(n_neighbors=5)
        model.fit(train_features, train_labels)
        predictions = model.predict(train_features)
        loss = np.mean((predictions - train_labels) ** 2)
        losses = [loss] * epochs  # 保持长度一致
        print(f'KNN Loss: {loss:.4f}')  # 打印损失值

    return losses


# 可视化多个模型的训练损失
def plot_multiple_model_losses(losses_dict, result_dir):
    plt.figure(figsize=(20, 12))
    # 损失曲线
    plt.subplot(2, 2, 1)
    plt.plot(svm_train_losses, label='SVM Training Loss')
    plt.title('Loss Curves')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    plt.subplot(2, 2, 2)
    plt.plot(rf_train_losses, label='Random Forest Training Loss')
    plt.title('Loss Curves')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.show()
    plt.savefig(os.path.join(result_dir, 'training_comparative_losses.png'), bbox_inches='tight')



def load_model(model_path, device):
    """加载训练好的最佳模型"""
    try:
        model = GNNModel()
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval()
        logging.info(f"成功加载模型：{model_path}")
        return model
    except Exception as e:
        logging.error(f"模型加载失败: {str(e)}")
        raise RuntimeError(f"模型加载失败: {str(e)}")

svm_train_losses = []
rf_train_losses = []
if __name__ == "__main__":
    result_dir = getPath() + '/result'
    os.makedirs(result_dir, exist_ok=True)

    # 配置统一日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(result_dir, 'training.log1.txt')),
            logging.StreamHandler()
        ]
    )

    # 数据加载与处理
    dataset = YouPPARyChooseBinaryDataset(root='data/')
    train_size = int(0.8 * len(dataset))
    train_data = dataset[:train_size]
    val_data = dataset[train_size:]

    train_loader = DataLoader(train_data, batch_size=2 ** 12, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=2 ** 12, shuffle=False)

    # 模型配置
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    gnn_model = load_model(os.path.join(result_dir, 'best_model.pth'), device)

    # 提取特征向量
    train_features = extract_features(gnn_model, train_loader, device)
    val_features = extract_features(gnn_model, val_loader, device)

    # 提取标签
    train_labels = np.array([data.y.numpy().flatten() for data in train_loader.dataset])
    val_labels = np.array([data.y.numpy().flatten() for data in val_loader.dataset])

    # 训练参数
    epochs = 400

    # 训练各模型
    logging.info("开始训练SVM模型...")
    svm_train_losses = train_sklearn_model("SVM", train_features, train_labels, epochs)

    logging.info("开始训练随机森林模型...")
    rf_train_losses = train_sklearn_model("RF", train_features, train_labels, epochs)

    logging.info("开始训练KNN模型...")
    knn_train_losses = train_sklearn_model("KNN", train_features, train_labels, epochs)

    # 假设GNN模型的训练损失已经存在，这里直接使用一个示例损失列表
    # 如果需要实际训练GNN模型，需要实现相应的训练逻辑
    # gnn_train_losses = [0.01 * (1 - i / epochs) for i in range(epochs)]  # 示例损失值

    # 收集所有模型的训练损失
    losses_dict = {
        'SVM': svm_train_losses,
        'Random Forest': rf_train_losses,
    }

    # 绘制多个模型的训练损失图
    plot_multiple_model_losses(losses_dict, result_dir)

    logging.info("训练完成，结果已保存")
