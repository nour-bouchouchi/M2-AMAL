from torch.utils.data import  DataLoader
import torch
import matplotlib.pyplot as plt
from utils import RNN, device, ForecastMetroDataset
from tqdm import tqdm
import numpy as np
import torch.nn as nn

# Nombre de stations utilisé
CLASSES = 10
#Longueur des séquences
LENGTH = 20
# Dimension de l'entrée (1 (in) ou 2 (in/out))
DIM_INPUT = 2
#Taille du batch
BATCH_SIZE = 32

PATH = "/home/ubuntu/Documents/Sorbonne/M2/M2-AMAL/TME4/data/"


matrix_train, matrix_test = torch.load(open(PATH+"hzdataset.pch", "rb"))
ds_train = ForecastMetroDataset(
    matrix_train[:, :, :CLASSES, :DIM_INPUT], length=LENGTH)
ds_test = ForecastMetroDataset(
    matrix_test[:, :, :CLASSES, :DIM_INPUT], length=LENGTH, stations_max=ds_train.stations_max)
data_train = DataLoader(ds_train, batch_size=BATCH_SIZE, shuffle=True)
data_test = DataLoader(ds_test, batch_size=BATCH_SIZE, shuffle=False)

#  TODO:  Question 3 : Prédiction de séries temporelles

dim_latent = 5
dim_output = CLASSES

pas = 2

epoch = 30
mse = nn.MSELoss()
model = RNN(DIM_INPUT, dim_latent, DIM_INPUT)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
liste_loss = []
liste_loss_test = []
for epoch in tqdm(range(epoch)):
    loss_epoch = 0
    accuracy_epoch = 0
    cpt = 0

    for x,_ in data_train:
        x = np.transpose(x, (1,0,2,3))
        x = x.reshape(x.shape[0], -1, DIM_INPUT)

        optimizer.zero_grad()

        loss = 0
        h = torch.zeros((  x.shape[1], dim_latent))
        for l in range(x.shape[0]-pas):
            h = model.one_step(x[l], h)
            ht = h
            pred = torch.zeros(( pas, x.shape[1], DIM_INPUT))
            for p in range(pas):
                ht = model.one_step(x[l+p+1], h)
                decode = model.decode(ht)
                pred[p] = decode        

            y = x[l+1:l+pas+1]

            loss += mse(decode, y)

        loss.backward()
        optimizer.step()
                    
        loss_epoch += loss.item()
        cpt += 1

    liste_loss.append(loss_epoch / cpt)


    accuracy_epoch = 0
    cpt = 0

    for x,_ in data_test:
        x = np.transpose(x, (1,0,2,3))
        x = x.reshape(x.shape[0], -1, DIM_INPUT)

        optimizer.zero_grad()

        loss = 0
        with torch.no_grad():
            h = torch.zeros((  x.shape[1], dim_latent))
            for l in range(x.shape[0]-pas):
                h = model.one_step(x[l], h)
                ht = h
                pred = torch.zeros(( pas, x.shape[1], DIM_INPUT))
                for p in range(pas):
                    ht = model.one_step(x[l+p+1], ht)
                    decode = model.decode(h)
                    pred[p] = decode        

                y = x[l+1:l+pas+1]

                loss += mse(decode, y)

        optimizer.step()
                    
        loss_epoch += loss.item()
        cpt += 1
    liste_loss_test.append(loss_epoch / cpt)



plt.figure()
plt.plot(np.arange(len(liste_loss)), liste_loss, label='Loss train', color='tab:orange')
plt.xlabel("Iterations")
plt.plot(np.arange(len(liste_loss_test)), liste_loss_test, label='Loss test', color='tab:blue')
plt.ylabel("Loss")
plt.title("Loss en train et en test")
plt.legend(loc='upper left')
plt.show()

