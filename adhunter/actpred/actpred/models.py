import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as Fun
from torchmetrics import MeanSquaredError, PearsonCorrCoef, SpearmanCorrCoef


class ResBlock(nn.Module):

    def __init__(self, hidden, kernel_size, dilation):
        super(ResBlock, self).__init__()
        self.bn_1 = nn.BatchNorm1d(num_features=hidden)
        self.relu_1 = nn.ReLU()
        self.conv_res = nn.Conv1d(in_channels=hidden,
                                  out_channels=hidden,
                                  kernel_size=kernel_size,
                                  padding="same",
                                  dilation=dilation)
        self.bn_2 = nn.BatchNorm1d(num_features=hidden)
        self.relu_2 = nn.ReLU()
        self.conv_block = nn.Conv1d(in_channels=hidden,
                                    out_channels=hidden,
                                    kernel_size=1,
                                    padding="same")

    def forward(self, X):
        out = self.bn_1(X)
        out = self.relu_1(out)
        out = self.conv_res(out)
        out = self.bn_2(out)
        out = self.relu_2(out)
        out = self.conv_block(out)
        out = out + X
        return out


class CNNBlock(nn.Module):

    def __init__(self, hidden, kernel_size):
        super(CNNBlock, self).__init__()
        self.bn_1 = nn.BatchNorm1d(num_features=hidden)
        self.relu_1 = nn.ReLU()
        self.conv_res = nn.Conv1d(in_channels=hidden,
                                  out_channels=hidden,
                                  kernel_size=kernel_size,
                                  padding="same")
        self.dropout = nn.Dropout(p=0.1)

    def forward(self, X):
        out = self.bn_1(X)
        out = self.relu_1(out)
        out = self.conv_res(out)
        out = self.dropout(out)
        return out


class PaddleCNN(nn.Module):

    def __init__(self, hidden=30, kernel_size=10, seq_len=53):
        super(PaddleCNN, self).__init__()
        self.emb = nn.Embedding(20, embedding_dim=hidden)
        self.conv_init = nn.Conv1d(in_channels=hidden,
                                   out_channels=hidden,
                                   kernel_size=kernel_size,
                                   padding="same")
        self.res_blocks = nn.ModuleList(
            [CNNBlock(hidden, kernel_size) for _ in range(9)])
        self.pool = nn.MaxPool1d(kernel_size=seq_len)
        self.lin1 = nn.Linear(hidden, 20)
        self.relu_1 = nn.ReLU()
        self.lin2 = nn.Linear(20, 20)
        self.relu_2 = nn.ReLU()
        self.lin3 = nn.Linear(20, 1)

    def forward(self, X):
        out = self.emb(X)
        out = out.transpose(2, 1)
        out = self.conv_init(out)
        for res_block in self.res_blocks:
            out = res_block(out)
        out = self.pool(out).squeeze()
        out = self.lin1(out)
        out = self.relu_1(out)
        out = self.lin2(out)
        out = self.relu_2(out)
        out = self.lin3(out)
        return out


class ActCNN(nn.Module):

    def __init__(self,
                 hidden,
                 kernel_size,
                 dilation,
                 num_res_blocks=3,
                 seq_len=40,
                 one_hot=True):
        super(ActCNN, self).__init__()
        self.one_hot_explicit = one_hot
        if self.one_hot_explicit:
            self.emb = nn.Linear(20, hidden, bias=False)
        else:
            self.emb = nn.Embedding(20, embedding_dim=hidden, dtype=torch.float32)
        self.conv_init = nn.Conv1d(in_channels=hidden,
                                   kernel_size=kernel_size,
                                   out_channels=hidden,
                                   padding="same")
        self.res_blocks = nn.ModuleList([
            ResBlock(hidden, kernel_size, dilation)
            for _ in range(num_res_blocks)
        ])
        self.pool = nn.MaxPool1d(kernel_size=seq_len)
        self.lin = nn.Linear(hidden, 1)

    def forward(self, X):
        if self.one_hot_explicit:
            X = self.encode(X).float()
        out = self.emb(X)
        out = out.transpose(2, 1)
        out = self.conv_init(out)
        for res_block in self.res_blocks:
            out = res_block(out)
        out = self.pool(out).squeeze()
        out = self.lin(out)
        return out
    
    def one_hot_custom(self, arr, num_classes:int):
        """Custom one hot encoding to allow encoding invalid indices 
        as all 0 vectors.

        Args:
            arr (torch.tensor): Array of token indices corresponding to desired one hot index. 
                Input value >= num_classes to encode all 0's.
            num_classes (int): _description_

        Returns:
            torch.tensor: one hot encoded array
        """        
        valid_mask = (arr >=0) & (arr < num_classes)
        if valid_mask.all():
            out = Fun.one_hot(arr, num_classes=num_classes)
        else:
            classes = range(num_classes)
            rows, cols = arr.shape
            out = torch.zeros((rows, cols, num_classes))
            for i in range(rows):
                for j in range(cols):
                    if arr[i, j] in classes:
                        out[i, j, arr[i, j]] = 1
                    out[i, j] = torch.zeros(num_classes)
        return out
    
    def encode(self, seq):
        seq_len, n_aas = 40, 20
        if type(seq) == torch.Tensor:
            if seq.shape[-1] == n_aas:
                # Do nothing if already proper shape for one hot encoded
                return seq.float()
            elif seq.shape[-1] == seq_len:
                # One hot encode if seq is a list of integer tokens of AAs
                seq = self.one_hot_custom(seq, num_classes=n_aas)
                return seq.float()
        elif type(seq) == str:
            alphabet="ACDEFGHIKLMNPQRSTVWY"
            aa_to_i = {aa:i for i, aa in enumerate(alphabet)}
            assert len(seq) == seq_len, f"Sequence must be seq_len AA!"
            seq = self.one_hot_custom(torch.tensor([[aa_to_i[i] for i in seq]], dtype=torch.int64), num_classes=n_aas)
            return seq.float()

class ActCNNSystem(pl.LightningModule):

    def __init__(self,
                 hidden,
                 kernel_size,
                 dilation,
                 num_res_blocks=3,
                 seq_len=40,
                 weight_decay=1e-2,
                 one_hot=True):
        super(ActCNNSystem, self).__init__()
        self.save_hyperparameters()
        self.wd = weight_decay
        self.model = ActCNN(hidden,
                            kernel_size,
                            dilation,
                            seq_len=seq_len,
                            num_res_blocks=num_res_blocks,
                            one_hot=one_hot)
        self.loss_fn = nn.MSELoss()

        self.rmse = MeanSquaredError(squared=False)
        self.pearsonr = PearsonCorrCoef()
        self.spearmanr = SpearmanCorrCoef()
        self.metrics = {
            "rmse": self.rmse,
            "pearsonr": self.pearsonr,
            "spearmanr": self.spearmanr
        }

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(),
                                     lr=1e-3,
                                     weight_decay=self.wd)
        return optimizer

    def training_step(self, batch, batch_idx):
        X, y = batch
        y_pred = self.model(X)
        loss = self.loss_fn(y_pred, y)
        return {
            "loss": loss,
            "y_target": y.view(-1),
            "y_pred": y_pred.detach().view(-1),
        }

    def training_epoch_end(self, train_step_outputs):
        y_preds = [d['y_pred'] for d in train_step_outputs]
        y_targets = [d['y_target'] for d in train_step_outputs]
        y_preds = torch.concat(y_preds)
        y_targets = torch.concat(y_targets)

        train_loss = self.metrics['rmse'](y_preds, y_targets)
        for metric_name, metric in self.metrics.items():
            metric_name = "train_" + metric_name
            self.log(metric_name, metric(y_preds, y_targets))
        return

    def validation_step(self, batch, batch_idx):
        X, y = batch
        y_pred = self.model(X)
        return (y_pred.view(-1), y.view(-1))

    def validation_epoch_end(self, val_step_outputs):
        y_preds, y_targets = zip(*val_step_outputs)
        y_preds = torch.concat(y_preds)
        y_targets = torch.concat(y_targets)

        val_loss = self.metrics['rmse'](y_preds, y_targets)
        self.log("val_loss", val_loss)
        for metric_name, metric in self.metrics.items():
            metric_name = "val_" + metric_name
            print(metric_name, metric(y_preds, y_targets).item(), flush=True)
            self.log(metric_name, metric(y_preds, y_targets))
        return val_loss
    
    
class ActCNNOneHot(nn.Module):
    """ADHunter model class. Similar to ActCNN, but with explicit 
    one-hot encoding fed into a linear layer to facilitate 
    interpretability.
    """
    def __init__(self,
                 hidden,
                 kernel_size,
                 dilation,
                 num_res_blocks=3,
                 seq_len=40):
        super(ActCNNOneHot, self).__init__()
        self.emb = nn.Linear(20, hidden, bias=False)
        self.conv_init = nn.Conv1d(in_channels=hidden,
                                   kernel_size=kernel_size,
                                   out_channels=hidden,
                                   padding="same")
        self.res_blocks = nn.ModuleList([
            ResBlock(hidden, kernel_size, dilation)
            for _ in range(num_res_blocks)
        ])
        self.pool = nn.MaxPool1d(kernel_size=seq_len)
        self.lin = nn.Linear(hidden, 1)

    def forward(self, X):
        # X = Fun.one_hot(X, num_classes=20).float()
        X = self.one_hot_custom(X, num_classes=20).float()
        out = self.emb(X)
        out = out.transpose(2, 1)
        out = self.conv_init(out)
        for res_block in self.res_blocks:
            out = res_block(out)
        out = self.pool(out).squeeze()
        out = self.lin(out)
        return out
    
    def one_hot_custom(self, arr, num_classes:int):
        """Custom one hot encoding to allow encoding invalid indices 
        as all 0 vectors.

        Args:
            arr (torch.tensor): Array of token indices corresponding to desired one hot index. 
                Input value >= num_classes to encode all 0's.
            num_classes (int): _description_

        Returns:
            torch.tensor: one hot encoded array
        """        
        valid_mask = (arr >=0) & (arr < num_classes)
        if valid_mask.all():
            out = Fun.one_hot(arr, num_classes=num_classes)
        else:
            classes = range(num_classes)
            rows, cols = arr.shape
            out = torch.zeros((rows, cols, num_classes))
            for i in range(rows):
                for j in range(cols):
                    if arr[i, j] in classes:
                        out[i, j, arr[i, j]] = 1
                    out[i, j] = torch.zeros(num_classes)
        return out


class ActCNNOneHotSystem(pl.LightningModule):
    """Wrapper for ActCNNOneHot model.
    """
    def __init__(self,
                 hidden,
                 kernel_size,
                 dilation,
                 num_res_blocks=3,
                 seq_len=40,
                 weight_decay=1e-2):
        super(ActCNNOneHotSystem, self).__init__()
        self.save_hyperparameters()
        self.wd = weight_decay
        self.model = ActCNNOneHot(hidden,
                            kernel_size,
                            dilation,
                            seq_len=seq_len,
                            num_res_blocks=num_res_blocks)
        self.loss_fn = nn.MSELoss()

        self.rmse = MeanSquaredError(squared=False)
        self.pearsonr = PearsonCorrCoef()
        self.spearmanr = SpearmanCorrCoef()
        self.metrics = {
            "rmse": self.rmse,
            "pearsonr": self.pearsonr,
            "spearmanr": self.spearmanr
        }

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(),
                                     lr=1e-3,
                                     weight_decay=self.wd)
        return optimizer

    def training_step(self, batch, batch_idx):
        X, y = batch
        y_pred = self.model(X)
        loss = self.loss_fn(y_pred, y)
        return {
            "loss": loss,
            "y_target": y.view(-1),
            "y_pred": y_pred.detach().view(-1),
        }

    def training_epoch_end(self, train_step_outputs):
        y_preds = [d['y_pred'] for d in train_step_outputs]
        y_targets = [d['y_target'] for d in train_step_outputs]
        y_preds = torch.concat(y_preds)
        y_targets = torch.concat(y_targets)

        train_loss = self.metrics['rmse'](y_preds, y_targets)
        for metric_name, metric in self.metrics.items():
            metric_name = "train_" + metric_name
            self.log(metric_name, metric(y_preds, y_targets))
        return

    def validation_step(self, batch, batch_idx):
        X, y = batch
        y_pred = self.model(X)
        return (y_pred.view(-1), y.view(-1))

    def validation_epoch_end(self, val_step_outputs):
        y_preds, y_targets = zip(*val_step_outputs)
        y_preds = torch.concat(y_preds)
        y_targets = torch.concat(y_targets)

        val_loss = self.metrics['rmse'](y_preds, y_targets)
        self.log("val_loss", val_loss)
        for metric_name, metric in self.metrics.items():
            metric_name = "val_" + metric_name
            print(metric_name, metric(y_preds, y_targets).item(), flush=True)
            self.log(metric_name, metric(y_preds, y_targets))
        return val_loss

class PaddleCNNSystem(pl.LightningModule):

    def __init__(self, weight_decay=1e-3):
        super(PaddleCNNSystem, self).__init__()
        self.save_hyperparameters()
        self.wd = weight_decay
        self.model = PaddleCNN()
        self.loss_fn = nn.MSELoss()

        self.rmse = MeanSquaredError(squared=False)
        self.pearsonr = PearsonCorrCoef()
        self.spearmanr = SpearmanCorrCoef()
        self.metrics = {
            "rmse": self.rmse,
            "pearsonr": self.pearsonr,
            "spearmanr": self.spearmanr
        }

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(),
                                     lr=1e-3,
                                     weight_decay=self.wd)
        return optimizer

    def training_step(self, batch, batch_idx):
        X, y = batch
        y_pred = self.model(X)
        loss = self.loss_fn(y_pred, y)
        return {
            "loss": loss,
            "y_target": y.view(-1),
            "y_pred": y_pred.detach().view(-1),
        }

    def training_epoch_end(self, train_step_outputs):
        y_preds = [d['y_pred'] for d in train_step_outputs]
        y_targets = [d['y_target'] for d in train_step_outputs]
        y_preds = torch.concat(y_preds)
        y_targets = torch.concat(y_targets)

        train_loss = self.metrics['rmse'](y_preds, y_targets)
        for metric_name, metric in self.metrics.items():
            metric_name = "train_" + metric_name
            self.log(metric_name, metric(y_preds, y_targets))
        return
    
    # def on_train_epoch_end(self, train_step_outputs):
    #     # Same as above, just renamed for PyTorch Lightning v2.0.0
    #     y_preds = [d['y_pred'] for d in train_step_outputs]
    #     y_targets = [d['y_target'] for d in train_step_outputs]
    #     y_preds = torch.concat(y_preds)
    #     y_targets = torch.concat(y_targets)

    #     train_loss = self.metrics['rmse'](y_preds, y_targets)
    #     for metric_name, metric in self.metrics.items():
    #         metric_name = "train_" + metric_name
    #         self.log(metric_name, metric(y_preds, y_targets))
    #     return

    def validation_step(self, batch, batch_idx):
        X, y = batch
        y_pred = self.model(X)
        return (y_pred.view(-1), y.view(-1))

    def validation_epoch_end(self, val_step_outputs):
        y_preds, y_targets = zip(*val_step_outputs)
        y_preds = torch.concat(y_preds)
        y_targets = torch.concat(y_targets)

        val_loss = self.metrics['rmse'](y_preds, y_targets)
        self.log("val_loss", val_loss)
        for metric_name, metric in self.metrics.items():
            metric_name = "val_" + metric_name
            print(metric_name, metric(y_preds, y_targets).item(), flush=True)
            self.log(metric_name, metric(y_preds, y_targets))
        return val_loss
    
    # def on_validation_epoch_end(self, val_step_outputs):
    #     # Same as above, just renamed for PyTorch Lightning v2.0.0
    #     y_preds, y_targets = zip(*val_step_outputs)
    #     y_preds = torch.concat(y_preds)
    #     y_targets = torch.concat(y_targets)

    #     val_loss = self.metrics['rmse'](y_preds, y_targets)
    #     self.log("val_loss", val_loss)
    #     for metric_name, metric in self.metrics.items():
    #         metric_name = "val_" + metric_name
    #         print(metric_name, metric(y_preds, y_targets).item(), flush=True)
    #         self.log(metric_name, metric(y_preds, y_targets))
    #     return val_loss
