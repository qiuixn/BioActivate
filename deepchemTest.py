import deepchem as dc
import pandas as pd
from deepchem.models import GraphConvModel
from deepchem.feat import MolGraphConvFeaturizer
from deepchem.splits import RandomSplitter
from deepchem.metrics import Metric
from deepchem.trans import NormalizationTransformer
from deepchem.feat.mol_graphs import ConvMol
from deepchem.feat import ConvMolFeaturizer

import numpy as np
from utils.util import *
print(np.__version__)

df = pd.read_csv(getPath()+"/data/rawData/data_PPAR-γ_filtered.csv", iterator=0)

# 特征化
featurizer = ConvMolFeaturizer()
X = featurizer.featurize(df['smiles'])
y = df['standard_value'].values
dataset = dc.data.NumpyDataset(X=X, y=y)

# 数据集划分
splitter = RandomSplitter()
train_dataset, test_dataset = splitter.train_test_split(dataset, frac_train=0.8)

# 数据预处理
transformers = [
    NormalizationTransformer(transform_y=True, dataset=train_dataset)
]
for transformer in transformers:
    train_dataset = transformer.transform(train_dataset)
    test_dataset = transformer.transform(test_dataset)

# 模型构建与训练
model = GraphConvModel(n_tasks=1, mode='regression')

# 记录训练过程中的损失值
losses = []

# 自定义训练循环
for epoch in range(5):
    # 训练一个 epoch
    loss = model.fit(train_dataset, nb_epoch=1)
    # 提取损失值
    if isinstance(loss, dict):
        loss_value = loss.get('loss')
    else:
        loss_value = loss
    losses.append(loss_value)
    print(f"Epoch {epoch + 1}: Loss = {loss_value}")
import matplotlib.pyplot as plt
# 训练结束后绘制图表
plt.figure(figsize=(10, 6))
plt.plot(losses, 'b', linewidth=1.5, label='Training Loss')
plt.title('Training Loss Curve', fontsize=14)
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Loss', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend()

# 自动调整坐标轴范围
ymin, ymax = np.min(losses)*0.9, np.max(losses)*1.1
plt.ylim(ymin, ymax)

# 保存并显示图像
plt.savefig('./training_loss.png', dpi=300, bbox_inches='tight')
plt.show()

# 模型评估
metric = Metric(dc.metrics.pearson_r2_score)
train_scores = model.evaluate(train_dataset, [metric], transformers)
test_scores = model.evaluate(test_dataset, [metric], transformers)
print("Training set score:", train_scores)
print("Test set score:", test_scores)

# 输出训练过程中记录的损失值
print("Losses during training:", losses)

# 模型应用
new_data = pd.DataFrame({'smiles': ['CCCO', 'NC']})
X_new = featurizer.featurize(new_data['smiles'])
new_dataset = dc.data.NumpyDataset(X=X_new)
predictions = model.predict(new_dataset)
print("Predictions:", predictions)