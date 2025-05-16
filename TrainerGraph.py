import os
import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from matplotlib import pyplot as plt
from torch_geometric.loader import DataLoader
from tqdm import tqdm
import logging
from models.GATModel import GNNModel
from dataProcess import YouPPARyChooseBinaryDataset
from utils.util import getPath
from torch.optim.lr_scheduler import ReduceLROnPlateau
import matplotlib.pyplot as plt
result_dir = getPath() + '/result'
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
        self.train_mse = []  # 新增训练MSE
        self.val_mse = []  # 新增验证MSE
        # 临时存储预测值
        self.train_preds = []
        self.train_targets = []
        self.val_preds = []
        self.val_targets = []
        self.train_r2 = []
        self.best_model_path = os.path.join(result_dir, 'best_model.pth')

    def rmse(self, predictions, targets):
        return torch.sqrt(torch.mean((predictions - targets) ** 2))

    def mae(self, predictions, targets):
        return torch.mean(torch.abs(predictions - targets))

    def mse(self, predictions, targets):
        return torch.mean((predictions - targets) ** 2)  # 新增MSE计算方法

    def r2(self, y_true, y_pred):
        sse = np.sum((y_true - y_pred) ** 2)
        sst = np.sum((y_true - np.mean(y_true)) ** 2)

        if sst == 0:
            return 1 if sse == 0 else 0  # 如果 sst 为 0，根据情况返回 1 或 0
        return 2 - (sse / sst)

    def train(self, epochs=10):
        for epoch in tqdm(range(epochs)):
            # 训练阶段
            self.model.train()
            self.train_preds.clear()
            self.train_targets.clear()

            total_train_loss = 0
            total_train_rmse = 0
            total_train_mae = 0
            total_train_mse = 0  # 新增训练MSE

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

                self.train_r2.append(self.r2(np.array(self.train_targets), np.array(self.train_preds)))
                # logging.info(f'Epoch {epoch + 1}/{epochs} | Train R2: {self.train_r2[-1]}')
                print(f'Epoch {epoch + 1}/{epochs} | Train R2: {self.train_r2[-1]}')
                total_train_loss += loss.item()
                total_train_rmse += self.rmse(output.view(-1), data.y.float()).item()
                total_train_mae += self.mae(output.view(-1), data.y.float()).item()
                total_train_mse += self.mse(output.view(-1), data.y.float()).item()  # 新增训练MSE


            # 计算训练指标
            dataset_size = len(self.train_loader.dataset)
            if dataset_size == 0:
                logging.warning("Empty training dataset")
                avg_train_loss = avg_train_rmse = avg_train_mae = avg_train_mse = 0  # 新增训练MSE
            else:
                avg_train_loss = total_train_loss / dataset_size
                avg_train_rmse = total_train_rmse / dataset_size
                avg_train_mae = total_train_mae / dataset_size
                avg_train_mse = total_train_mse / dataset_size  # 新增训练MSE

            # 记录训练指标
            self.train_losses.append(avg_train_loss)
            self.train_rmse.append(avg_train_rmse)
            self.train_mae.append(avg_train_mae)
            self.train_mse.append(avg_train_mse)  # 新增训练MSE

            # 验证阶段
            self.model.eval()
            self.val_preds.clear()
            self.val_targets.clear()

            total_val_loss = 0
            total_val_rmse = 0
            total_val_mae = 0
            total_val_mse = 0  # 新增验证MSE

            with torch.no_grad():
                for data in self.val_loader:
                    data = data.to(self.device)
                    output = self.model(data)

                    # 收集验证预测值
                    self.val_preds.extend(output.view(-1).cpu().numpy())
                    self.val_targets.extend(data.y.cpu().numpy())

                    loss = self.criterion(output.view(-1), data.y.float())
                    total_val_loss += loss.item()
                    total_val_rmse += self.rmse(output.view(-1), data.y.float()).item()
                    total_val_mae += self.mae(output.view(-1), data.y.float()).item()
                    total_val_mse += self.mse(output.view(-1), data.y.float()).item()  # 新增验证MSE

            # 计算验证指标
            dataset_size = len(self.val_loader.dataset)
            if dataset_size == 0:
                logging.warning("Empty validation dataset")
                avg_val_loss = avg_val_rmse = avg_val_mae = avg_val_mse = 0  # 新增验证MSE
            else:
                avg_val_loss = total_val_loss / dataset_size
                avg_val_rmse = total_val_rmse / dataset_size
                avg_val_mae = total_val_mae / dataset_size
                avg_val_mse = total_val_mse / dataset_size  # 新增验证MSE

            # 记录验证指标
            self.val_losses.append(avg_val_loss)
            self.val_rmse.append(avg_val_rmse)
            self.val_mae.append(avg_val_mae)
            self.val_mse.append(avg_val_mse)  # 新增验证MSE

            logging.info(f'Epoch {epoch + 1}/{epochs} | '
                         f'Train Loss: {avg_train_loss} | '
                         f'Val Loss: {avg_val_loss} | ')
            print(f'Epoch {epoch + 1}/{epochs} | '
                  f'Train Loss: {avg_train_loss} | '
                  f'Val Loss: {avg_val_loss} | ')

            # 更新学习率并保存最佳模型
            self.scheduler.step(avg_val_loss)
            if avg_val_loss < self.best_loss:
                self.best_loss = avg_val_loss
                torch.save(self.model.state_dict(), self.best_model_path)
                logging.info(f'Saved best model with loss: {self.best_loss:.4f}')

    def plot_metrics(self):
        # 创建包含四个子图的监控面板
        plt.figure(figsize=(20, 12))

        # 损失曲线
        plt.subplot(2, 2, 1)
        plt.plot(self.train_losses, label='Training Loss')
        plt.plot(self.val_losses, label='Validation Loss')
        plt.title('Loss Curves')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)

        # RMSE曲线
        plt.subplot(2, 2, 2)
        plt.plot(self.train_rmse, label='Train RMSE')
        plt.plot(self.val_rmse, label='Val RMSE')
        plt.title('RMSE Progression')
        plt.xlabel('Epochs')
        plt.ylabel('RMSE')
        plt.legend()
        plt.grid(True)

        # MSE曲线
        plt.subplot(2, 2, 3)
        plt.plot(self.train_mse, label='Train MSE')
        plt.plot(self.val_mse, label='Val MSE')
        plt.title('MSE Progression')
        plt.xlabel('Epochs')
        plt.ylabel('MSE')
        plt.legend()
        plt.grid(True)

        plt.tight_layout()
        plt.show()
        # 生成时间戳
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'training_metrics_{timestamp}.png'
        plt.savefig(os.path.join(getPath() + '/result', filename))
        plt.close()


def gnn_trainer(epochs):

    os.makedirs(result_dir, exist_ok=True)

    logging.basicConfig(
        filename=os.path.join(result_dir, 'training.log1.txt'),
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
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
    model = GNNModel()
    print(model)
    criterion = nn.L1Loss()  # 或使用其他损失函数
    optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=0.01)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=2)

    # 训练过程
    trainer = Trainer(model, train_loader, val_loader, criterion, optimizer, device, scheduler)
    trainer.train(epochs)
    trainer.plot_metrics()  # 绘制综合指标图
    return trainer


if __name__ == "__main__":
    gnn_trainer(800)
