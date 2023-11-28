import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import optuna
from optuna.visualization import plot_contour
from optuna.visualization import plot_edf
from optuna.visualization import plot_intermediate_values
from optuna.visualization import plot_optimization_history
from optuna.visualization import plot_parallel_coordinate
from optuna.visualization import plot_param_importances
from optuna.visualization import plot_rank
from optuna.visualization import plot_slice
from optuna.visualization import plot_timeline

from torch.utils.data import Subset
import datetime
import argparse

from torch.functional import norm
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader,random_split,TensorDataset
from pathlib import Path
from datamaestro import prepare_dataset
import time
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from tp7 import *

"""
def objective(trial):
    from tp7 import MLP3_dropout_batchNorm, train, NUM_CLASSES, INPUT_DIM
    iterations = 200
    dims = [100, 100, 100]

    norm_type = trial.suggest_categorical('normalization', ["identity", "batchnorm", "layernorm"])
    normalization = norm_type
    dropouts = [trial.suggest_loguniform('dropout_p%d' % ix, 1e-2, 0.5) for ix in range(len(dims))]

    l2 = trial.suggest_uniform('l2', 0, 1)
    l1 = trial.suggest_uniform('l1', 0, 1)

    model = MLP3_dropout_batchNorm(INPUT_DIM, NUM_CLASSES, dims)
    return train(iterations, model, l1, l2)

study = optuna.create_study()
study.optimize(objective, n_trials=20)
print(study.best_params)
"""


BATCH_SIZE = 311
TRAIN_RATIO = 0.05


class LitMnistData(pl.LightningDataModule):

    def __init__(self,batch_size=BATCH_SIZE,train_ratio=TRAIN_RATIO):
        super().__init__()
        self.dim_in = None
        self.dim_out = None
        self.batch_size = batch_size
        self.train_ratio = train_ratio

    def prepare_data(self):
        ### Do not use "self" here.
        prepare_dataset("com.lecun.mnist")

    def setup(self,stage=None):
        ds = prepare_dataset("com.lecun.mnist")
        if stage =="fit" or stage is None:
            # Si on est en phase d'apprentissage
            shape = ds.train.images.data().shape
            self.dim_in = shape[1]*shape[2]
            self.dim_out = len(set(ds.train.labels.data()))
            ds_train = TensorDataset(torch.tensor(ds.train.images.data()).view(-1,self.dim_in).float()/255., torch.tensor(ds.train.labels.data()).long())
            train_length = int(shape[0]*self.train_ratio)
            self.mnist_train, self.mnist_val, = random_split(ds_train,[train_length,shape[0]-train_length])
        if stage == "test" or stage is None:
            # en phase de test
            self.mnist_test= TensorDataset(torch.tensor(ds.test.images.data()).view(-1,self.dim_in).float()/255., torch.tensor(ds.test.labels.data()).long())

    def train_dataloader(self):
        return DataLoader(self.mnist_train,batch_size=self.batch_size)
    def val_dataloader(self):
        return DataLoader(self.mnist_val,batch_size=self.batch_size)
    def test_dataloader(self):
        return DataLoader(self.mnist_test,batch_size=self.batch_size)



def objective(trial):


    dim_in =  784
    dim_out = 10

    param = {
        "l": trial.suggest_int('l', 10, 100)  ,
        "lr": trial.suggest_float('lr', 1e-5, 1e-1, log=True)
    }
    model = MLP3_dropout_batchNorm(dim_in, param["l"], dim_out).to(device)
    optimizer = optim.SGD(model.parameters(), lr=param["lr"])
    epochs = 10 
    _, _, _, acc_val = train(sub_train_loader, val_loader, model, optimizer, epochs, device, None, param_reg_l1=None)

    return acc_val[-1]


if __name__ == "__main__": 

    LOG_PATH = "/tmp/runs/lightning_logs"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data = LitMnistData()
    data.prepare_data()

    data.setup()
    train_loader = data.train_dataloader()
    val_loader  = data.val_dataloader()
    test_loader = data.test_dataloader()

    sub_train_length = int(len(train_loader.dataset) * TRAIN_RATIO)
    sub_train_set = Subset(train_loader.dataset, range(sub_train_length))
    sub_train_loader = DataLoader(sub_train_set, batch_size=train_loader.batch_size, shuffle=True)

    pruner = optuna.pruners.MedianPruner()
    study = optuna.create_study(direction='maximize', pruner=pruner)
    study.optimize(objective, n_trials=100)



    best_trial = study.best_trial
    best_params = best_trial.params
    print("Best trial:")
    print(f"Accuracy: {best_trial.value}") 
    print(f"Params: {best_params}")

    plot_optimization_history(study)
    plot_intermediate_values(study)
    plot_parallel_coordinate(study)
    plot_contour(study)
    plot_slice(study)
    plot_param_importances(study)
    plot_edf(study)
    plot_rank(study)
    plot_timeline(study)