import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.loader import DataLoader
from tqdm import tqdm
import logging
from models.GATModel import GNNModel
from dataProcess import YouPPARyChooseBinaryDataset
from utils.util import getPath
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_squared_log_error
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt

# 定义 DNN 模型
class DNNModel(nn.Module):
    def __init__(self, input_dim):
        super(DNNModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)

    def forward(self, x):
        x = nn.relu(self.fc1(x))
        x = nn.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# 训练 DNN 模型
def train_dnn_model(model, train_loader, criterion, optimizer, device, epochs=10):
    model.train()
    for epoch in tqdm(range(epochs)):
        total_loss = 0
        for data in train_loader:
            x, y = data.x.to(device), data.y.to(device).float().view(-1, 1)
            optimizer.zero_grad()
            output = model(x)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(train_loader.dataset)
        logging.info(f'Epoch {epoch + 1}/{epochs} | DNN Model Train Loss: {avg_loss:.4f}')

# 训练 sklearn 模型
def train_sklearn_model(model, train_features, train_labels):
    model.fit(train_features, train_labels)

# 评估 sklearn 模型
def evaluate_sklearn_model(model, test_features, test_labels):
    predictions = model.predict(test_features)
    mse = mean_squared_error(test_labels, predictions)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(test_labels, predictions)
    return mse, rmse, mae

# 定义 Trainer 类
class Trainer:
    def __init__(self, model, train_loader, val_loader, criterion, optimizer, device, scheduler):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.scheduler = scheduler
        self.best_loss = float('inf')
        # 初始化指标存储
        self.train_losses = []
        self.val_losses = []
        self.train_rmse = []
        self.val_rmse = []
        self.train_mae = []
        self.val_mae = []
        self.train_mse = []
        self.val_mse = []
        # 临时存储预测值
        self.train_preds = []
        self.train_targets = []
        self.val_preds = []
        self.val_targets = []

        self.best_model_path = os.path.join(result_dir, 'best_model.pth')

    def rmse(self, predictions, targets):
        return torch.sqrt(torch.mean((predictions - targets) ** 2))

    def mae(self, predictions, targets):
        return torch.mean(torch.abs(predictions - targets))

    def mse(self, predictions, targets):
        return torch.mean((predictions - targets) ** 2)

    def train(self, epochs=10):
        for epoch in tqdm(range(epochs)):
            # 训练阶段
            self.model.train()
            self.train_preds.clear()
            self.train_targets.clear()

            total_train_loss = 0
            total_train_rmse = 0
            total_train_mae = 0
            total_train_mse = 0

            try:
                for data in self.train_loader:
                    data = data.to(self.device)
                    self.optimizer.zero_grad()
                    output = self.model(data)

                    # 收集预测值和真实值
                    self.train_preds.extend(output.view(-1).detach().cpu().numpy())
                    self.train_targets.extend(data.y.cpu().numpy())

                    loss = self.criterion(output.view(-1), data.y.float())
                    loss.backward()
                    self.optimizer.step()

                    total_train_loss += loss.item()
                    total_train_rmse += self.rmse(output.view(-1), data.y.float()).item()
                    total_train_mae += self.mae(output.view(-1), data.y.float()).item()
                    total_train_mse += self.mse(output.view(-1), data.y.float()).item()

            except Exception as e:
                logging.error(f"Training error: {e}")
                break

            # 计算训练指标
            dataset_size = len(self.train_loader.dataset)
            if dataset_size == 0:
                logging.warning("Empty training dataset")
                avg_train_loss = avg_train_rmse = avg_train_mae = avg_train_mse = 0
            else:
                avg_train_loss = total_train_loss / dataset_size
                avg_train_rmse = total_train_rmse / dataset_size
                avg_train_mae = total_train_mae / dataset_size
                avg_train_mse = total_train_mse / dataset_size

            # 记录训练指标
            self.train_losses.append(avg_train_loss)
            self.train_rmse.append(avg_train_rmse)
           