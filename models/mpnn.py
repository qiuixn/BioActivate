"""
Author: Derek van Tilborg -- TU/e -- 23-05-2022

Message Passing Neural Network for molecular property prediction using continuous kernel-based convolution
(edge-conditioned convolution) [1] and global graph pooling using a graph multiset transformer [2] instead of
the Set2Set method used in [1]. I added dropout to make a more robust

[1] Gilmer et al. (2017). Neural Message Passing for Quantum Chemistry
[2] Baek et al. (2021). Accurate Learning of Graph Representations with Graph Multiset Pooling

"""
import os

import torch
import torch.nn.functional as F
from torch.nn import GRU, Linear, ReLU, Sequential, Dropout
from torch_geometric.nn import NNConv, GraphMultisetTransformer

RANDOM_SEED = 42

def graphs_to_loader(x: List[Data], y: List[float], batch_size: int = 64, shuffle: bool = False):
    """ Turn a list of graph objects and a list of labels into a Dataloader """
    for graph, label in zip(x, y):
        graph.y = torch.tensor(label)

    return DataLoader(x, batch_size=batch_size, shuffle=shuffle)

class GNN:
    """ Base GNN class that takes care of training, testing, predicting for all graph-based methods """
    def __init__(self):
        self.train_losses = []
        self.val_losses = []
        self.epoch = 0
        self.epochs = 100
        self.save_path = os.path.join('.', 'best_model.pkl')

        self.model = None
        self.device = None
        self.loss_fn = None
        self.optimizer = None


    def train(self, x_train: List[Data], y_train: List[float], x_val: List[Data] = None, y_val: List[float] = None,
              early_stopping_patience: int = None, epochs: int = None, print_every_n: int = 100):
        """ Train a graph neural network.

        :param x_train: (List[Data]) a list of graph objects for training
        :param y_train: (List[float]) a list of bioactivites for training
        :param x_val: (List[Data]) a list of graph objects for validation
        :param y_val: (List[float]) a list of bioactivites for validation
        :param epochs: (int) train the model for n epochs
        :param print_every_n: (int) printout training progress every n epochs
        """
        if epochs is None:
            epochs = self.epochs
        train_loader = graphs_to_loader(x_train, y_train)
        patience = None if early_stopping_patience is None else 0

        for epoch in range(epochs):

            # If we reached the end of our patience, load the best model and stop training
            if patience is not None and patience >= early_stopping_patience:

                if print_every_n < epochs:
                    print('Stopping training early')
                try:
                    with open(self.save_path, 'rb') as handle:
                        self.model = pickle.load(handle)

                    os.remove(self.save_path)
                except Warning:
                    print('Could not load best model, keeping the current weights instead')

                break

            # As long as the model is still improving, continue training
            else:
                loss = self._one_epoch(train_loader)
                self.train_losses.append(loss)

                val_loss = 0
                if x_val is not None:
                    val_pred = self.predict(x_val)
                    val_loss = self.loss_fn(squeeze_if_needed(val_pred), torch.tensor(y_val))
                self.val_losses.append(val_loss)

                self.epoch += 1

                # Pickle model if its the best
                if val_loss <= min(self.val_losses):
                    with open(self.save_path, 'wb') as handle:
                        pickle.dump(self.model, handle, protocol=pickle.HIGHEST_PROTOCOL)
                    patience = 0
                else:
                    patience += 1

                if self.epoch % print_every_n == 0:
                    print(f"Epoch {self.epoch} | Train Loss {loss} | Val Loss {val_loss}")

    def _one_epoch(self, train_loader):
        """ Perform one forward pass of the train data through the model and perform backprop

        :param train_loader: Torch geometric data loader with training data
        :return: loss
        """
        # Enumerate over the data
        for idx, batch in enumerate(train_loader):

            # Move batch to gpu
            batch.to(self.device)

            # Reset gradients
            self.optimizer.zero_grad()

            # Forward pass
            y_hat = self.model(batch.x.float(), batch.edge_index, batch.edge_attr.float(), batch.batch)

            # Calculating the loss and gradients
            loss = self.loss_fn(squeeze_if_needed(y_hat), squeeze_if_needed(batch.y))
            if not loss > 0:
                print(idx)

            # Calculate gradients
            loss.backward()

            # Update weights
            self.optimizer.step()

        return loss

    def test(self,  x_test: List[Data], y_test: List[float]):
        """ Perform testing

        :param x_test: (List[Data]) a list of graph objects for testing
        :param y_test: (List[float]) a list of bioactivites for testing
        :return: A tuple of two 1D-tensors (predicted, true)
        """
        data_loader = graphs_to_loader(x_test, y_test, shuffle=False)
        y_pred, y = [], []
        with torch.no_grad():
            for batch in data_loader:
                batch.to(self.device)
                y_hat = self.model(batch.x.float(), batch.edge_index, batch.edge_attr.float(), batch.batch)
                y_hat = squeeze_if_needed(y_hat).tolist()
                if type(y_hat) is list:
                    y_pred.extend(y_hat)
                    y.extend(squeeze_if_needed(batch.y).tolist())
                else:
                    y_pred.append(y_hat)
                    y.append(squeeze_if_needed(batch.y).tolist())

        return torch.tensor(y_pred), torch.tensor(y)

    def predict(self, x, batch_size: int = 32):
        """ Predict bioactivity on molecular graphs

        :param x_test: (List[Data]) a list of graph objects for testing
        :param batch_size: (int) batch size for the prediction loader
        :return: A 1D-tensors of predicted values
        """
        loader = DataLoader(x, batch_size=batch_size, shuffle=False)
        y_pred = []
        with torch.no_grad():
            for batch in loader:
                batch.to(self.device)
                y_hat = self.model(batch.x.float(), batch.edge_index, batch.edge_attr.float(), batch.batch)
                y_hat = squeeze_if_needed(y_hat).tolist()
                if type(y_hat) is list:
                    y_pred.extend(y_hat)
                else:
                    y_pred.append(y_hat)


        return torch.tensor(y_pred)

    def __repr__(self):
        return 'Basic Graph Neural Network Class'

class MPNN(GNN):
    def __init__(self, node_in_feats: int = 37, node_hidden: int = 64, edge_in_feats: int = 6,
                 edge_hidden: int = 128, message_steps: int = 3, dropout: float = 0.2,
                 transformer_heads: int = 8, transformer_hidden: int = 128, seed: int = RANDOM_SEED,
                 fc_hidden: int = 64, n_fc_layers: int = 1, lr: float = 0.0005, epochs: int = 300, *args, **kwargs):
        super().__init__()

        self.model = MPNNmodel(node_in_feats=node_in_feats, node_hidden=node_hidden, edge_in_feats=edge_in_feats,
                               edge_hidden=edge_hidden, message_steps=message_steps, dropout=dropout,
                               transformer_heads=transformer_heads, transformer_hidden=transformer_hidden, seed=seed,
                               fc_hidden=fc_hidden, n_fc_layers=n_fc_layers)

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.loss_fn = torch.nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.epochs = epochs
        self.name = 'MPNN'

        # Move the whole model to the gpu
        self.model = self.model.to(self.device)

    def __repr__(self):
        return f"{self.model}"


class MPNNmodel(torch.nn.Module):
    def __init__(self, node_in_feats: int = 37, node_hidden: int = 64, edge_in_feats: int = 6,
                 edge_hidden: int = 128, message_steps: int = 3, dropout: float = 0.2,
                 transformer_heads: int = 8, transformer_hidden: int = 128, seed: int = RANDOM_SEED,
                 fc_hidden: int = 64, n_fc_layers: int = 1, *args, **kwargs):

        # Init parent
        super().__init__()
        torch.manual_seed(seed)

        self.seed = seed
        self.node_hidden = node_hidden
        self.messsage_steps = message_steps
        self.node_in_feats = node_in_feats

        # Layer to project node features to hidden features
        self.project_node_feats = Sequential(Linear(node_in_feats, node_hidden), ReLU())

        # The 'learnable message function'
        edge_network = Sequential(Linear(edge_in_feats, edge_hidden), ReLU(),
                                  Linear(edge_hidden, node_hidden * node_hidden))

        # edge-conditioned convolution layer
        self.gnn_layer = NNConv(in_channels=node_hidden,
                                out_channels=node_hidden,
                                nn=edge_network,
                                aggr='add')

        # The GRU as used in [1]
        self.gru = GRU(node_hidden, node_hidden)

        # Global pooling using a transformer
        self.transformer = GraphMultisetTransformer(in_channels=node_hidden,
                                                    hidden_channels=transformer_hidden,
                                                    out_channels=fc_hidden,
                                                    num_heads=transformer_heads)

        # fully connected layer(s)
        self.fc = torch.nn.ModuleList()
        for k in range(n_fc_layers):
            self.fc.append(Linear(fc_hidden, fc_hidden))

        self.dropout = Dropout(dropout)
        self.lin2 = torch.nn.Linear(fc_hidden, 1)

    def forward(self, x, edge_index, edge_attr, batch):

        # Project node features to node hidden dimensions, which are also used as h_0 for the GRU
        node_feats = self.project_node_feats(x)
        hidden_feats = node_feats.unsqueeze(0)

        # Perform n message passing steps using edge-conditioned convolution as pass it through a GRU
        for _ in range(self.messsage_steps):
            node_feats = F.relu(self.gnn_layer(node_feats, edge_index, edge_attr))
            node_feats, hidden_feats = self.gru(node_feats.unsqueeze(0), hidden_feats)
            node_feats = node_feats.squeeze(0)

        # perform global pooling using a multiset transformer to get graph-wise hidden embeddings
        out = self.transformer(node_feats, batch, edge_index)

        # Apply a fully connected layer.
        for k in range(len(self.fc)):
            out = F.relu(self.fc[k](out))
            out = self.dropout(out)

        out = self.lin2(out)

        return out
