import logging
import os
from torch.nn.modules.pooling import MaxPool1d
logging.basicConfig(level=logging.INFO)

import heapq
from pathlib import Path
import gzip

from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import sentencepiece as spm
import numpy as np
import datetime
import matplotlib.pyplot as plt

from tp8_preprocess import TextDataset

# Utiliser tp8_preprocess pour générer le vocabulaire BPE et
# le jeu de donnée dans un format compact

# --- Configuration

# Taille du vocabulaire
vocab_size = 1000
#MAINDIR = Path(__file__).parent
current_file = os.path.abspath(os.getcwd())
MAINDIR = Path(current_file).parent

# Chargement du tokenizer

tokenizer = spm.SentencePieceProcessor()
tokenizer.Load(f"wp{vocab_size}.model")
ntokens = len(tokenizer)

def loaddata(mode):
    with gzip.open(f"{mode}-{vocab_size}.pth", "rb") as fp:
        return torch.load(fp)


test = loaddata("test")
train = loaddata("train")
TRAIN_BATCHSIZE=500
TEST_BATCHSIZE=500


# --- Chargements des jeux de données train, validation et test

val_size = 1000
train_size = len(train) - val_size
train, val = torch.utils.data.random_split(train, [train_size, val_size])

logging.info("Datasets: train=%d, val=%d, test=%d", train_size, val_size, len(test))
logging.info("Vocabulary size: %d", vocab_size)
train_iter = torch.utils.data.DataLoader(train, batch_size=TRAIN_BATCHSIZE, collate_fn=TextDataset.collate)
val_iter = torch.utils.data.DataLoader(val, batch_size=TEST_BATCHSIZE, collate_fn=TextDataset.collate)
test_iter = torch.utils.data.DataLoader(test, batch_size=TEST_BATCHSIZE, collate_fn=TextDataset.collate)


def eval(model, loader, nb_epoch, writer, device):
    criterion = nn.CrossEntropyLoss()  
    loss_values = []
    accuracy_values = [] 

    model.eval()
    write_tensorboard = nb_epoch//20

    for epoch in tqdm(range(nb_epoch)):
        epoch_loss = []
        epoch_accuracy = []
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(loader):
                data, target = data.to(device), target.to(device)
                logits = model(data)
                loss = criterion(logits, target)
                
                epoch_loss.append(loss.item())
                epoch_accuracy.append((logits.argmax(1).detach().to('cpu') == target.detach().to('cpu')).sum().item() / len(target))

            
            loss_values.append(np.mean(epoch_loss))
            accuracy_values.append(np.mean(epoch_accuracy))

            if write_tensorboard != 0 and epoch % write_tensorboard == 0:

                writer.add_scalar('Loss validation', np.mean(epoch_loss), epoch)
            
                for name, weight in model.named_parameters():
                    writer.add_histogram(name, weight, epoch)
                    writer.add_histogram(f'{name}.grad', weight.grad, epoch)

                entropie = criterion(logits,target)
                writer.add_histogram('Entropy validation', entropie, epoch)
                
                writer.add_scalar('Accuracy validation', np.mean(epoch_accuracy), epoch)
    
    return loss_values, accuracy_values


def train(model, optimizer, train_loader, val_loader, nb_epoch, writer, device):
    criterion = nn.CrossEntropyLoss()  
    loss_values = []
    accuracy_values = [] 

    model.train()
    write_tensorboard = nb_epoch//20

    for epoch in tqdm(range(nb_epoch)):
        epoch_loss = []
        epoch_accuracy = []

        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()

            logits = model(data)
            loss = criterion(logits, target)
            
            loss.backward()
            optimizer.step()

            epoch_loss.append(loss.item())
            epoch_accuracy.append((logits.argmax(1).detach().to('cpu') == target.detach().to('cpu')).sum().item() / len(target))

        
        loss_values.append(np.mean(epoch_loss))
        accuracy_values.append(np.mean(epoch_accuracy))

        if write_tensorboard != 0 and epoch % write_tensorboard == 0:

            writer.add_scalar('Loss train', np.mean(epoch_loss), epoch)
        
            for name, weight in model.named_parameters():
                writer.add_histogram(f'{name}', weight, epoch)
                writer.add_histogram(f'{name}.grad', weight.grad, epoch)

            entropie = criterion(logits,target)
            writer.add_histogram('Entropy train', entropie, epoch)
            
            writer.add_scalar('Accuracy train', np.mean(epoch_accuracy), epoch)


        loss_eval, acc_eval = eval(model, val_loader, nb_epoch, writer, device)
        
    return loss_values, accuracy_values, loss_eval, acc_eval

def test(loader, model, device):
    acc = 0
    model.eval()
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(loader):
            data, target = data.to(device), target.to(device)
            output = model(data)
            acc += (output.argmax(1).detach().to('cpu') == target.detach().to('cpu')).sum().item() / len(target)
 
    acc /= len(loader)
    return acc

class CNN(nn.Module):
    def __init__(self, vocab_size, in_channels_list, out_channels_list, kernel_sizes_list, nb_classes):
        super(CNN, self).__init__()
        
        self.emb = nn.Embedding(vocab_size, in_channels_list[0])
        
        self.conv = nn.ModuleList(
            nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size, stride=1, padding=1), 
                nn.ReLU(),
                nn.MaxPool1d(kernel_size) if i < len(in_channels_list) - 1 else nn.Identity()
            )
            
            for i, (in_channels, out_channels, kernel_size) in enumerate(zip(in_channels_list, out_channels_list, kernel_sizes_list))
        )

        self.classifier = nn.Linear(out_channels_list[-1], nb_classes)
    
    def forward(self, x):
        nb_batch = x.shape[0]
        #print("x : ",x.shape)
        
        x = self.emb(x).permute(0,2,1)
        #print("emb : ",x.shape)

        for convol in self.conv : 
            x = convol(x) 
            #print("conv : ", x.shape)
        #print("fin conv : ",x.shape)

        res = [self.classifier(x[:,:,i]).unsqueeze(0) for i in range(x.shape[2])]
        res = torch.cat(res, dim=0)
        #print("res : ", res.shape)

        logits, _ = torch.max(res, dim=0)
        #print("logits : ", logits.shape)
        return logits


class Trivial_algo():
    def __init__(self, train_labels):
        unique_classes, counts = torch.unique(train_labels, return_counts=True)
        self.majority_class = unique_classes[counts.argmax()]

    def predict(self, input_size):
        return torch.full((input_size,), self.majority_class)


#  TODO: 

if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(device)

    ######## Algo trivial : classe majoritaire ########
    train_labels = torch.tensor([label for _, label in train_iter.dataset])
    majority_baseline = Trivial_algo(train_labels)
    val_labels = torch.tensor([label for _, label in val_iter.dataset])
    majority_class_predictions = majority_baseline.predict(len(val_iter.dataset))
    majority_class_accuracy = torch.sum(majority_class_predictions == val_labels).item() / len(val_labels)
    print("Majority Class Baseline Accuracy on Validation Set: {:.4f}".format(majority_class_accuracy))
    
    ######## CNN ##########

    #vocab_size = 1000
    in_channels_list = [16, 100, 200]
    out_channels_list = [100, 200, 300]
    kernel_sizes_list = [3, 3, 3]
    nb_classes = 3
    nb_epoch = 10

    model = CNN(vocab_size, in_channels_list, out_channels_list, kernel_sizes_list, nb_classes).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    writer = SummaryWriter("tp8/"+datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

    loss_values, accuracy_values, loss_eval, acc_eval = train(model, optimizer, train_iter, val_iter, nb_epoch, writer, device)

    plt.figure()
    plt.plot(loss_values,label="Loss en train")
    plt.plot(loss_eval,label="Loss en validation")
    plt.legend()
    plt.show()

    plt.figure()
    plt.plot(accuracy_values,label="Accuracy en train")
    plt.plot(acc_eval,label="Accuracy en validation")
    plt.legend()
    plt.show()

    acc_test = test(test_iter, model, device)

    print("Accuracy en train : ", accuracy_values[-1])
    print("Accuracy en validation : ", acc_eval[-1])
    print("Accuracy en test : ", acc_test)