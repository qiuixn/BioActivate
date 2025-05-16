# -*- coding: utf-8 -*-
import re
import matplotlib.pyplot as plt
from utils.util import getPath

pattern = r"Epoch (\d+)/\d+.*(Training|Validation) Loss: (\d+\.\d+)"

# 初始化存储结构
epoch_data = {}
current_epoch = None


with open(getPath()+"/result/log.txt", "r", encoding='gbk', errors='ignore') as f:
    for line in f:
        # 使用正则匹配关键信息
        match = re.search(pattern, line)
        if match:
            epoch = int(match.group(1))
            loss_type = match.group(2)
            loss_value = float(match.group(3))

            # 初始化epoch条目
            if epoch not in epoch_data:
                epoch_data[epoch] = {"train": None, "val": None}

            # 记录损失值
            if loss_type == "Training":
                epoch_data[epoch]["train"] = loss_value
            elif loss_type == "Validation":
                epoch_data[epoch]["val"] = loss_value

# 转换数据为有序列表
epochs = sorted(epoch_data.keys())
train_losses = [epoch_data[e]["train"] for e in epochs]
val_losses = [epoch_data[e]["val"] for e in epochs]

# 可视化设置
plt.figure(figsize=(12, 6), dpi=150)
plt.plot(epochs, train_losses, 'b', lw=1.5, label='Training Loss')
plt.plot(epochs, val_losses, 'r', lw=1.5, label='Validation Loss')

# 图表装饰
plt.title('Training and Validation Loss Curves', fontsize=14)
plt.xlabel('Epochs', fontsize=12)
plt.ylabel('Loss', fontsize=12)
plt.grid(True, alpha=0.3)
plt.legend()
def plot_train_val_losses(train_losses,val_losses):
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss Over Epochs')
    plt.savefig('training_validation_loss_xiaorong1.png', bbox_inches='tight')
    plt.legend()
    plt.show()
print(train_losses[:850])
plot_train_val_losses(train_losses,val_losses)
# # 显示并保存图表
# plt.tight_layout()
# plt.savefig('training_validation_loss.png', bbox_inches='tight')
# plt.show()