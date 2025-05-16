import deepchem as dc
import pandas as pd
from deepchem.models import MPNNModel
from deepchem.feat import MolGraphConvFeaturizer
from deepchem.splits import RandomSplitter
from deepchem.metrics import Metric
from deepchem.trans import NormalizationTransformer
from deepchem.feat.mol_graphs import ConvMol
from deepchem.feat import ConvMolFeaturizer
from utils.util import *


df = pd.read_csv(getPath()+"/data/rawData/data_PPAR-γ_filtered.csv", iterator=0)

# 特征化
featurizer = ConvMolFeaturizer()
X = featurizer.featurize(df['smiles'])
y = df['standard_value'].values
dataset = dc.data.NumpyDataset(X=X, y=y)

# import tensorflow as tf
# # 限制TensorFlow线程池
# tf.config.threading.set_intra_op_parallelism_threads(4)  # 单个操作内部线程数
# tf.config.threading.set_inter_op_parallelism_threads(2)  # 操作间并行线程数
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
model = MPNNModel(n_tasks=1, mode='regression')

# 记录训练过程中的损失值
all_losses = []

# 自定义训练循环
# 训练一个 epoch
losses = model.fit(train_dataset, nb_epoch=1000,all_losses=all_losses)
print(all_losses)
print(losses)


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