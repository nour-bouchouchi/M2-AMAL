import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from utils import RNN, device, SampleMetroDataset
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


# Nombre de stations utilisé
CLASSES = 10
#Longueur des séquences
LENGTH = 20
# Dimension de l'entrée (1 (in) ou 2 (in/out))
DIM_INPUT = 2
#Taille du batch
BATCH_SIZE = 32

PATH = "/home/ubuntu/Documents/Sorbonne/M2/M2-AMAL/TME4/data/"


matrix_train, matrix_test = torch.load(open(PATH+"hzdataset.pch","rb"))
ds_train = SampleMetroDataset(matrix_train[:, :, :CLASSES, :DIM_INPUT], length=LENGTH)
ds_test = SampleMetroDataset(matrix_test[:, :, :CLASSES, :DIM_INPUT], length = LENGTH, stations_max = ds_train.stations_max)
data_train = DataLoader(ds_train,batch_size=BATCH_SIZE,shuffle=True)
data_test = DataLoader(ds_test, batch_size=BATCH_SIZE,shuffle=False)

dim_latent = 5
dim_output = CLASSES

epoch = 40
ce = nn.CrossEntropyLoss()
model = RNN(DIM_INPUT, dim_latent, dim_output)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
liste_loss = []
liste_accuracy = []
liste_loss_test = []
liste_accuracy_test = []
for epoch in tqdm(range(epoch)):
    #print("epoch : ", epoch)
    loss_epoch = 0
    accuracy_epoch = 0
    cpt = 0
    e = 0

    for x, y in data_train:
        x = np.transpose(x, (1,0,2))
        optimizer.zero_grad()
        yhat = model(x)
        decode = model.decode(yhat[-1])
        
        loss = ce(decode, y) 
        loss.backward()
        optimizer.step()
        
    
        predicted_labels = torch.argmax(decode, dim=1)
        accuracy = (torch.sum(torch.eq(predicted_labels, y)).item() )/len(y)
        accuracy_epoch += accuracy
        
        loss_epoch += loss.item()
        cpt += 1

    liste_accuracy.append(accuracy_epoch / cpt)
    liste_loss.append(loss_epoch / cpt)


    loss_epoch = 0
    accuracy_epoch = 0
    cpt = 0
    with torch.no_grad() : 
        for x, y in data_test:
            x = np.transpose(x, (1,0,2))
            yhat = model(x)
            decode = model.decode(yhat[-1])
            
            loss = ce(decode, y) 
            predicted_labels = torch.argmax(decode, dim=1)
            accuracy = (torch.sum(torch.eq(predicted_labels, y)).item() )/len(y)
            accuracy_epoch += accuracy
            
            loss_epoch += loss.item()
            cpt += 1

    liste_accuracy_test.append(accuracy_epoch / cpt)
    liste_loss_test.append(loss_epoch / cpt)




plt.figure()
plt.plot(np.arange(len(liste_loss)), liste_loss, label='Loss train', color='tab:orange')
plt.xlabel("Iterations")
plt.plot(np.arange(len(liste_loss_test)), liste_loss_test, label='Loss test', color='tab:blue')
plt.ylabel("Loss")
plt.title("Loss en train et en test")
plt.legend(loc='upper left')
plt.show()

plt.figure()
plt.plot(np.arange(len(liste_accuracy)), liste_accuracy, label='Accuracy train', color='tab:orange')
plt.xlabel("Iterations")
plt.ylabel("Accuracy")
plt.plot(np.arange(len(liste_accuracy_test)), liste_accuracy_test, label='Accuracy test', color='tab:blue')
plt.title("Accuracy and  Loss en test")
plt.legend(loc='upper left')
plt.show()