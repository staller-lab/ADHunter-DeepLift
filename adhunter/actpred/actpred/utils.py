import torch
from pytorch_lightning.callbacks.progress import TQDMProgressBar
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn import preprocessing
from sklearn.metrics import roc_curve
import numpy as np


def get_threshold(y_bin_test, y_test_hat):
    fpr, tpr, thresholds = roc_curve(y_bin_test, y_test_hat)
    gmeans = np.sqrt(tpr * (1 - fpr))
    ix = np.argmax(gmeans)
    best_thresh = thresholds[ix]
    return best_thresh


def get_stratified_split(X,
                         y_bin,
                         y_cont,
                         as_tensor=True,
                         return_onehot=False):

    splitter = StratifiedShuffleSplit(n_splits=1,
                                      train_size=0.8,
                                      random_state=0)
    train_index, val_test_index = list(splitter.split(X, y_bin))[0]

    X_train = X[train_index]
    y_bin_train = y_bin[train_index]
    y_cont_train = y_cont[train_index]

    X_val_test = X[val_test_index]
    y_bin_val_test = y_bin[val_test_index]
    y_cont_val_test = y_cont[val_test_index]

    splitter = StratifiedShuffleSplit(n_splits=1,
                                      train_size=0.5,
                                      random_state=0)
    val_index, test_index = list(splitter.split(X_val_test, y_bin_val_test))[0]
    X_val = X_val_test[val_index]
    y_bin_val = y_bin_val_test[val_index]
    y_cont_val = y_cont_val_test[val_index]

    X_test = X_val_test[test_index]
    y_bin_test = y_bin_val_test[test_index]
    y_cont_test = y_cont_val_test[test_index]

    if return_onehot:
        enc = preprocessing.OneHotEncoder()
        enc.fit(X)
        X_train_one_hot, X_val_one_hot, X_test_one_hot = map(
            enc.transform, (X_train, X_val, X_test))
        # Convert to dense arrays
        X_train_one_hot = X_train_one_hot.toarray().astype(np.float32)
        X_val_one_hot = X_val_one_hot.toarray().astype(np.float32)
        X_test_one_hot = X_test_one_hot.toarray().astype(np.float32)        

        if as_tensor:
            X_train = torch.tensor(X_train)
            X_train_one_hot = torch.tensor(X_train_one_hot)
            y_bin_train = torch.tensor(y_bin_train)
            y_cont_train = torch.tensor(y_cont_train)

            X_val = torch.tensor(X_val)
            X_val_one_hot = torch.tensor(X_val_one_hot)
            y_bin_val = torch.tensor(y_bin_val)
            y_cont_val = torch.tensor(y_cont_val)

            X_test = torch.tensor(X_test)
            X_test_one_hot = torch.tensor(X_test_one_hot)
            y_bin_test = torch.tensor(y_bin_test)
            y_cont_test = torch.tensor(y_cont_test)
        return (X_train, X_train_one_hot, y_bin_train, y_cont_train), \
            (X_val, X_val_one_hot, y_bin_val, y_cont_val), \
            (X_test, X_test_one_hot, y_bin_test, y_cont_test)
    else:
        if as_tensor:
            X_train = torch.tensor(X_train)
            y_bin_train = torch.tensor(y_bin_train)
            y_cont_train = torch.tensor(y_cont_train)

            X_val = torch.tensor(X_val)
            y_bin_val = torch.tensor(y_bin_val)
            y_cont_val = torch.tensor(y_cont_val)

            X_test = torch.tensor(X_test)
            y_bin_test = torch.tensor(y_bin_test)
            y_cont_test = torch.tensor(y_cont_test)
        return (X_train, y_bin_train, y_cont_train), \
            (X_val, y_bin_val, y_cont_val), \
            (X_test, y_bin_test, y_cont_test), \
            (val_index, test_index)


class EmacsProgressBar(TQDMProgressBar):

    def init_train_tqdm(self):
        from tqdm import tqdm
        bar = tqdm(desc="Training",
                   initial=self.train_batch_idx,
                   position=(2 * self.process_position))
        return bar

    def init_validation_tqdm(self):
        from tqdm import tqdm
        bar = tqdm(desc="Validating",
                   position=(2 * self.process_position),
                   disable=True)
        return bar
