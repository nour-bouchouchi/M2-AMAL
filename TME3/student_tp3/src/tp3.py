from pathlib import Path
import os
import torch
from torchvision.utils import make_grid
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import datetime
import matplotlib.pyplot as plt
from datamaestro import prepare_dataset


################################################
################ DATASET #######################
################################################

class MonDataset ( Dataset ) :
    def __init__ ( self , data, label ) :
        super(MonDataset, self).__init__()
        self._data = data/255.0
        self._label = label
        
    def __getitem__ ( self , index ) :
        """ retourne un couple ( exemple , label ) correspondant à l'index """
        return self._data[index], self._label[index]   

    def __len__( self ) :
        """ renvoie la taille du jeu de donnée """
        return self._data.shape[0]
    

################################################
############### AUTOENCODEUR ###################
################################################
class Autoencodeur(nn.Module) :
    def __init__ ( self, n_x , n_lat) :
        super(Autoencodeur, self).__init__()
        self._W = nn.Parameter( torch.randn(n_lat,n_x))
        self._biais_encodeur = nn.Parameter(torch.randn(n_lat))
        self._biais_decodeur = nn.Parameter(torch.randn(n_x))

    def encode(self, x):
        enc = F.linear(x, self._W, self._biais_encodeur)
        return F.relu(enc)

    def decode(self, x):
        dec = F.linear(x, self._W.T, self._biais_decodeur)
        return F.sigmoid(dec)

    def forward(self, x): 
        return self.decode(self.encode(x))

class Autoencodeur2(nn.Module) :
    def __init__ ( self, n_x , n_lat1, n_lat2) :
        super(Autoencodeur2, self).__init__()
        self._W1 = nn.Parameter(torch.randn(n_lat1,n_x))
        self._W2 = nn.Parameter(torch.randn(n_lat2,n_lat1))
        self._biais_encodeur1 = nn.Parameter(torch.randn(n_lat1))
        self._biais_encodeur2 = nn.Parameter(torch.randn(n_lat2))
        self._biais_decodeur1 = nn.Parameter(torch.randn(n_lat1))
        self._biais_decodeur2 = nn.Parameter(torch.randn(n_x))

    def encode(self, x):
        enc1 = F.tanh(F.linear(x, self._W1, self._biais_encodeur1))
        enc2 = F.relu(F.linear(enc1, self._W2, self._biais_encodeur2))                    
        return enc2

    def decode(self, x):
        dec1 = F.tanh(F.linear(x, self._W2.T, self._biais_decodeur1))
        dec2 = F.sigmoid(F.linear(dec1, self._W1.T, self._biais_decodeur2))
        return dec2

    def forward(self, x): 
        return self.decode(self.encode(x))
    
################################################
#################### STATE #####################
################################################

class State:
    def __init__(self, model, optimizer):
        self.model = model
        self.optimizer = optimizer
        self.epoch, self.iteration = 0, 0



################################################
#################### MAIN ######################
################################################

if __name__ == '__main__':

    ds = prepare_dataset("com.lecun.mnist");
    train_images, train_labels = ds.train.images.data(), ds.train.labels.data()
    test_images, test_labels =  ds.test.images.data(), ds.test.labels.data()
    
    


    # Tensorboard : rappel, lancer dans une console tensorboard --logdir runs
    writer = SummaryWriter("runs/runs"+datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    # Pour visualiser
    # Les images doivent etre en format Channel (3) x Hauteur x Largeur
    images = torch.tensor(train_images[0:8]).unsqueeze(1).repeat(1,3,1,1).double()/255.
    # Permet de fabriquer une grille d'images
    images = make_grid(images)
    # Affichage avec tensorboard
    writer.add_image(f'samples', images, 0)


    nb_train = train_images.shape[0]
    nb_test = test_images.shape[0]

    taille_img = train_images.shape[1]

    train_images = np.reshape(train_images, (nb_train, -1))
    test_images = np.reshape(test_images, (nb_test, -1))


    batch_size = 40
    latent_dim = 20
    latent_dim2 = 10

    mydataset = MonDataset(train_images, train_labels)
    train_loader = DataLoader(dataset=mydataset, batch_size=batch_size, shuffle=True)


    #auto = Autoencodeur(train_images.shape[1], latent_dim) 
    auto = Autoencodeur2(train_images.shape[1], latent_dim,latent_dim2 )
    auto = auto.double()
    mse = nn.MSELoss(reduction='mean')

    savepath = Path("model.pth")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if savepath.is_file():
        with savepath.open('rb') as fp:
            state = torch.load(fp)
    else:
        model =  auto  
        model = model.double()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)  

        state = State(model, optimizer)

    liste_loss = []
    for epoch in range(state.epoch, 10):
        print("epoch : ", epoch)
        loss_epoch = 0
        cpt = 0
        for x, y in train_loader:
            state.optimizer.zero_grad()
            x = x.to(device)
            xhat = state.model(x)
            loss = mse(xhat, x) 
            loss.backward()
            state.optimizer.step()
            state.iteration += 1
            loss_epoch+=loss.item()
            cpt+=1
        liste_loss.append(loss_epoch/cpt)

        with savepath.open('wb') as fp:
            state.epoch = epoch + 1
            torch.save(state, fp)

    plt.figure()
    plt.plot(np.arange(len(liste_loss)),liste_loss)
    plt.title("Loss  batch")
    plt.xlabel("Iterations")
    plt.ylabel("loss")
    plt.show()

    img = test_images[0]
    plt.imshow(img.reshape(taille_img, taille_img))
    plt.show()
    img_decode = state.model(torch.tensor(img, dtype = torch.float64))
    plt.imshow(img_decode.view(taille_img, taille_img).detach())
    plt.show()

