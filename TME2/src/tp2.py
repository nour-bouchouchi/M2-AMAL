import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import accuracy_score
## Installer datamaestro et datamaestro-ml pip install datamaestro datamaestro-ml
#import datamaestro
#from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split



writer = SummaryWriter()

#colnames, datax, datay = data.data()
#datax = torch.tensor(datax,dtype=torch.float)
#datay = torch.tensor(datay,dtype=torch.float).reshape(-1,1)
#data = datamaestro.prepare_dataset("edu.uci.boston")



def init_data():
    # Générer des données linéaires en 5 dimensions
    torch.manual_seed(42)
    num_samples = 1000
    num_features = 5

    # Créer des poids pour la relation linéaire
    true_weights = torch.tensor([2.0, -1.0, 0.5, 3.0, -2.0])

    # Générer des données aléatoires pour les caractéristiques (datax)
    datax = torch.randn(num_samples, num_features, dtype=torch.float64)

    # Générer des données cibles (datay) en utilisant la relation linéaire avec un biais
    true_bias = torch.tensor(1.0, dtype=torch.float64)
    datay = torch.mm(datax, true_weights.view(num_features, 1).to(torch.float64)) + true_bias

    # Ajouter du bruit gaussien aux données y
    noise = torch.randn(datay.shape, dtype=torch.float64)
    datay += noise * 0.2  # Ajout de bruit gaussien avec un écart-type de 0.2

    return datax, datay


datax, datay = init_data()

########################################################
################### question 1 #########################
########################################################

nb_epoch = 100
eps = 5e-2
####################################################################
########################## Cas batch ###############################
####################################################################
print("BATCH : ")

linear = torch.nn.Linear(5, 1, dtype=torch.float64)
mse = torch.nn.MSELoss()


liste_loss = []
for i in range(nb_epoch):

    yhat = linear(datax)
    loss = mse(yhat,datay)
    
    liste_loss.append(loss.item())

    loss_backward = loss.backward()

    with torch.no_grad():
        for param in linear.parameters():
            param -= eps * param.grad
    
    linear.zero_grad()



plt.figure()
plt.plot(np.arange(nb_epoch),liste_loss)
plt.title("Loss  batch")
plt.xlabel("Iterations")
plt.ylabel("loss")
plt.show()





####################################################################
########################## Cas batch ###############################
####################################################################
print("MINI BATCH : ")

linear = torch.nn.Linear(5, 1, dtype=torch.float64)
mse = torch.nn.MSELoss()


liste_loss = []
size_batch = 50

for i in range(nb_epoch):
    ind = torch.randperm(size_batch)
    batch_x = datax[ind]
    batch_y = datay[ind]

    yhat = linear(batch_x)
    loss = mse(yhat,batch_y)
    
    liste_loss.append(loss.item())

    loss_backward = loss.backward()

    with torch.no_grad():
        for param in linear.parameters():
            param -= eps * param.grad
    
    linear.zero_grad()



plt.figure()
plt.plot(np.arange(nb_epoch),liste_loss)
plt.title("Loss mini-batch 50")
plt.xlabel("Iterations")
plt.ylabel("loss")
plt.show()


########################################################
################### question 2 #########################
########################################################
print("QUESTION 2")
model = torch.nn.Sequential(
    torch.nn.Linear(5, 2, dtype=torch.float64),
    torch.nn.Tanh(),
    torch.nn.Linear(2, 1, dtype=torch.float64)
)

mse = torch.nn.MSELoss()

optimizer = torch.optim.SGD(model.parameters(), lr=eps)


liste_loss = []
for i in range(nb_epoch):

    yhat = model(datax)
    loss = mse(yhat,datay)
    
    liste_loss.append(loss.item())

    optimizer.zero_grad()
    loss_backward = loss.backward()
    optimizer.step()
    

plt.figure()
plt.plot(np.arange(nb_epoch),liste_loss)
plt.title("Loss batch")
plt.xlabel("Iterations")
plt.ylabel("loss")
plt.show()



