import string
import unicodedata
import torch
import sys
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset,DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm
from utils import RNN, device
import numpy as np

#Taille du batch
BATCH_SIZE = 32

PATH = "/home/ubuntu/Documents/Sorbonne/M2/M2-AMAL/TME4/data/"

## Liste des symboles autorisés
LETTRES = string.ascii_letters + string.punctuation+string.digits+' '
## Dictionnaire index -> lettre
id2lettre = dict(zip(range(1,len(LETTRES)+1),LETTRES))
id2lettre[0]='' ##NULL CHARACTER
## Dictionnaire lettre -> index
lettre2id = dict(zip(id2lettre.values(),id2lettre.keys()))


def normalize(s):
    """ Nettoyage d'une chaîne de caractères. """
    return ''.join(c for c in unicodedata.normalize('NFD', s) if  c in LETTRES)

def string2code(s):
    """ Transformation d'une chaîne de caractère en tenseur d'indexes """
    return torch.tensor([lettre2id[c] for c in normalize(s)])

def code2string(t):
    """ Transformation d'une liste d'indexes en chaîne de caractères """
    if type(t) !=list:
        t = t.tolist()
    return ''.join(id2lettre[i] for i in t)

class TrumpDataset(Dataset):
    def __init__(self,text,maxsent=None,maxlen=None):
        """  Dataset pour les tweets de Trump
            * text : texte brut
            * maxsent : nombre maximum de phrases.
            * maxlen : longueur maximale des phrases.
        """
        maxlen = maxlen or sys.maxsize
        full_text = normalize(text)
        self.phrases = [p[:maxlen].strip()+"." for p in full_text.split(".") if len(p)>0]
        if maxsent is not None:
            self.phrases=self.phrases[:maxsent]
        self.MAX_LEN = max([len(p) for p in self.phrases])

    def __len__(self):
        return len(self.phrases)
    
    def __getitem__(self,i):
        t = string2code(self.phrases[i])
        t = torch.cat([torch.zeros(self.MAX_LEN-t.size(0),dtype=torch.long),t])
        return t[:-1],t[1:]



#  TODO: 

data_trump = DataLoader(TrumpDataset(open(PATH+"trump_full_speech.txt","rb").read().decode(),maxlen=100), batch_size= BATCH_SIZE
, shuffle=True)


def one_hot(x, num_classes):
    one_hot = torch.nn.functional.one_hot(x, num_classes=num_classes)
    return one_hot.to(torch.float)
    


dim_input = len(LETTRES)+1
dim_latent = 50
dim_output = 100


epoch = 10
embeding = nn.Linear(dim_input, dim_input)
rnn = RNN(dim_input, dim_latent, dim_input)

params = list(rnn.parameters()) + list(embeding.parameters())

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(params, lr=1e-3, weight_decay=1e-5)


liste_loss = []
liste_loss_test = []
for epoch in tqdm(range(epoch)):
    
    liste_loss_tmp = []

    for x,y in data_trump:
        x_one_hot = one_hot(x, dim_input)
        x_emb = embeding(x_one_hot)

        y_one_hot = one_hot(y, dim_input)

        optimizer.zero_grad()

        liste_h  = rnn(x_emb) 
        yhat = rnn.decode(liste_h, "many-to-many")

        loss = criterion(yhat, y_one_hot)
        loss.backward()
        optimizer.step()
        
        with torch.no_grad():
            liste_loss_tmp.append(loss.item())
        
    liste_loss.append(np.mean(liste_loss_tmp))
    
plt.figure()
plt.plot(np.arange(len(liste_loss)), liste_loss, label='Loss train', color='tab:orange')
plt.xlabel("Iterations")
plt.title("Loss en train et en test")
plt.legend(loc='upper left')
plt.show()



debut = 'hello'
sentence = debut

debut = string2code(debut)
debut = one_hot(debut, dim_input)
debut = embeding(debut)

h_i = rnn(debut)
h_i = torch.stack(h_i)
yhat = rnn.decode(h_i,type='many-to-many')
yhat = yhat.softmax(dim=2)

x_i = torch.multinomial(yhat[0,0,:],1)
x_i = one_hot(x_i, dim_input)
#x_i = embeding(x_i)
sentence += code2string([torch.argmax(x_i).item()])

for _ in range(dim_output):
    h_i = rnn.one_step(x_i,h_i)
    yhat = rnn.decode(h_i,type='many-to-one').softmax(dim=2)
    x_i = torch.multinomial(yhat[0,0,:],1)
    x_i = one_hot(x_i, dim_input)
   # x_i = embeding(x_i)
    sentence += code2string([torch.argmax(x_i).item()])

print(sentence)

