import logging
logging.basicConfig(level=logging.INFO)

import os
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import click

from datamaestro import prepare_dataset

from torch.utils.data import Subset
import datetime
import argparse

from torch.functional import norm
import torch.nn as nn
from torch.utils.data import DataLoader,random_split,TensorDataset
from pathlib import Path
from datamaestro import prepare_dataset
import time
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
import numpy as np



# Ratio du jeu de train à utiliser
TRAIN_RATIO = 0.05
BATCH_SIZE = 311
TRAIN_RATIO = 0.05


def store_grad(var):
    """Stores the gradient during backward

    For a tensor x, call `store_grad(x)`
    before `loss.backward`. The gradient will be available
    as `x.grad`

    """
    def hook(grad):
        var.grad = grad
    var.register_hook(hook)
    return var


#  TODO:  Implémenter


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



class MLP3(nn.Module):
    def __init__(self, dim_in, l, dim_out):
        super().__init__()
        self.model = nn.Sequential(nn.Linear(dim_in, l), nn.ReLU(), nn.Linear(l, l), nn.ReLU(), nn.Linear(l, dim_out))

    def forward(self, x):
        x = self.model(x)
        return x


class MLP3_Dropout(nn.Module):
    def __init__(self, dim_in, l, dim_out):
        super().__init__()
        self.model = nn.Sequential(nn.Linear(dim_in, l), 
                                   nn.ReLU(), 
                                   nn.Dropout(0.1),
                                   nn.Linear(l, l), 
                                   nn.BatchNorm1d(l),
                                   nn.ReLU(), 
                                   nn.Dropout(0.1),
                                   nn.Linear(l, dim_out))

    def forward(self, x):
        x = self.model(x)
        return x

class MLP3_dropout_batchNorm(nn.Module):
    def __init__(self, dim_in, l, dim_out):
        super().__init__()
        self.model = nn.Sequential(nn.Linear(dim_in, l), 
                                   nn.BatchNorm1d(l),
                                   nn.ReLU(), 
                                   nn.Dropout(0.1),
                                   nn.Linear(l, l), 
                                   nn.BatchNorm1d(l),
                                   nn.ReLU(), 
                                   nn.Dropout(0.1),
                                   nn.Linear(l, dim_out))

    def forward(self, x):
        x = self.model(x)
        return x


class MLP3_dropout_layerNorm(nn.Module):
    def __init__(self, dim_in, l, dim_out):
        super().__init__()
        self.model = nn.Sequential(nn.Linear(dim_in, l), 
                                   nn.LayerNorm(l),
                                   nn.ReLU(), 
                                   nn.Dropout(0.1),
                                   nn.Linear(l, l), 
                                   nn.LayerNorm(l),
                                   nn.ReLU(), 
                                   nn.Dropout(0.1),
                                   nn.Linear(l, dim_out))

    def forward(self, x):
        x = self.model(x)
        return x



def train(train_loader, test_loader, model, optimizer, epoch, device, writer, param_reg_l1=None):
    l_train = []
    l_val = []
    a_train = []
    a_val = []
    
    criterion = nn.CrossEntropyLoss()  

    write_tensorboard = epoch//20

    for e in tqdm(range(epoch)):
        loss_train = []
        loss_val = []
        acc_train = []
        acc_val = []

        ##################
        ###### Train #####
        ##################

        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
           
            # Régularisation L1 
            if param_reg_l1 != None : 
                l1_reg = torch.tensor(0.0, requires_grad=True).to(device)
                for param in model.parameters():
                    l1_reg = l1_reg + torch.norm(param, 1)
                loss = loss + param_reg_l1 * l1_reg

            loss.backward()
            optimizer.step()
            
            loss_train.append(loss.item())
            acc_train.append((output.argmax(1).detach().to('cpu') == target.detach().to('cpu')).sum().item() / len(target))
        
        l_train.append(np.mean(loss_train))
        a_train.append(np.mean(acc_train))

        if write_tensorboard != 0 and e % write_tensorboard == 0:
 


            writer.add_scalar('Loss train', np.mean(loss_train), e)
        
        
            for name, weight in model.named_parameters():
                writer.add_histogram(name, weight, e)
                writer.add_histogram(f'{name}.grad', weight.grad, e)

            entropie = criterion(output,target)
            writer.add_histogram('Entropy train', entropie, e)

         
            writer.add_scalar('Accuracy train', np.mean(acc_train), e)

        ##################
        ###### Eval ######
        ##################

        model.eval()
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(test_loader):
                data, target = data.to(device), target.to(device)
                output = model(data)
                loss = criterion(output, target)
                
                loss_val.append(loss.item())
                acc_val.append((output.argmax(1).detach().to('cpu') == target.detach().to('cpu')).sum().item() / len(target))

        l_val.append(np.mean(loss_val))
        a_val.append(np.mean(acc_val))

        if write_tensorboard != 0 and e % write_tensorboard == 0:
 

            writer.add_scalar('Loss validation', np.mean(loss_val), e)
        
        
            for name, weight in model.named_parameters():
                writer.add_histogram(name, weight, e)
                writer.add_histogram(f'{name}.grad', weight.grad, e)

            entropie = criterion(output,target)
            writer.add_histogram('Entropy validation', entropie, e)

         
            writer.add_scalar('Accuracy validation', np.mean(acc_val), e)
    
    return l_train, l_val, a_train, a_val


def test(test_loader, model, device):
    acc = 0
    model.eval()
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)
            output = model(data)
            acc += (output.argmax(1).detach().to('cpu') == target.detach().to('cpu')).sum().item() / len(target)
 
    acc /= len(test_loader)
    return acc

    
if __name__ == "__main__": 
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(device)


    data = LitMnistData()
    data.prepare_data()

    data.setup()
    train_loader = data.train_dataloader()
    val_loader  = data.val_dataloader()
    test_loader = data.test_dataloader()

    sub_train_length = int(len(train_loader.dataset) * TRAIN_RATIO)
    sub_train_set = Subset(train_loader.dataset, range(sub_train_length))
    sub_train_loader = DataLoader(sub_train_set, batch_size=train_loader.batch_size, shuffle=True)


    dim_in = 784
    dim_out = 10
    train_length = 50000

    ### Modèle MLP3 ####
    epoch = 1000
    dim_latent = 100
    model = MLP3(dim_in, dim_latent, dim_out).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    writer = SummaryWriter("tp7/"+datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    train(sub_train_loader, val_loader, model, optimizer, epoch, device, writer)

    ### Régularisation l1/l2 ###
    l2_regularization = 0.01
    param_regularization_l1 = 1e-5
    writer_reg = SummaryWriter("tp7_reg/"+datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    model_reg = MLP3(dim_in, dim_latent, dim_out).to(device)
    optimizer_reg = torch.optim.SGD(model_reg.parameters(), lr=0.1, weight_decay=l2_regularization)
    train(sub_train_loader, test_loader, model_reg, optimizer_reg, epoch, device, writer_reg, param_reg_l1=param_regularization_l1)

    ### Dropout ###
    writer_reg_dropout = SummaryWriter("tp7_reg_dropout/"+datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    model_reg_dropout = MLP3_Dropout(dim_in, dim_latent, dim_out).to(device)
    optimizer_reg = torch.optim.SGD(model_reg_dropout.parameters(), lr=0.1, weight_decay=l2_regularization)
    train(sub_train_loader, test_loader, model_reg_dropout, optimizer_reg, epoch, device, writer_reg_dropout, param_reg_l1=param_regularization_l1)

    ### BatchNorm ###
    writer_reg_dropout_batchnorm = SummaryWriter("tp7_reg_dropout_batchnorm/"+datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    model_reg_dropout_batchnorm = MLP3_dropout_batchNorm(dim_in, dim_latent, dim_out).to(device)
    optimizer_reg = torch.optim.SGD(model_reg_dropout_batchnorm.parameters(), lr=0.1, weight_decay=l2_regularization)
    train(sub_train_loader, test_loader, model_reg_dropout_batchnorm, optimizer_reg, epoch, device, writer_reg_dropout_batchnorm, param_reg_l1=param_regularization_l1)

    ### LayerNorm ###
    writer_reg_dropout_layernorm = SummaryWriter("tp7_reg_dropout_layernorm/"+datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    model_reg_dropout_layernorm = MLP3_dropout_layerNorm(dim_in, dim_latent, dim_out).to(device)
    optimizer_reg = torch.optim.SGD(model_reg_dropout_layernorm.parameters(), lr=0.1, weight_decay=l2_regularization)
    train(sub_train_loader, test_loader, model_reg_dropout_layernorm, optimizer_reg, epoch, device, writer_reg_dropout_layernorm, param_reg_l1=param_regularization_l1)
