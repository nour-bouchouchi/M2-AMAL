
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from textloader import *
from generate import *
from tqdm import tqdm
import matplotlib.pyplot as plt
import datetime
import numpy as np

#  TODO: 

def maskedCrossEntropy(output: torch.Tensor, target: torch.LongTensor, padcar: int):
    """
    :param output: Tenseur length x batch x output_dim,
    :param target: Tenseur length x batch
    :param padcar: index du caractere de padding
    """
    #  TODO:  Implémenter maskedCrossEntropy sans aucune boucle, la CrossEntropy qui ne prend pas en compte les caractères de padding.
    mask = (target != padcar)
    loss = torch.nn.functional.cross_entropy(output.permute(0,2,1), target, reduction='none')
    return torch.mean(loss * mask)


class RNN(nn.Module):
    def __init__(self, dim_input, dim_latent, dim_output, dim_emb):
        super().__init__()
        self._dim_input = dim_input
        self._dim_latent = dim_latent
        self._dim_emb = dim_emb
        
        self.lin_i = nn.Linear(dim_emb, dim_latent, bias=False)
        self.lin_h = nn.Linear(dim_latent, dim_latent, bias=True)
        self.lin_d = nn.Linear(dim_latent, dim_output, bias=True)
        self._tanh = nn.Tanh() 

    def one_step(self,x,h):
        return self._tanh(self.lin_i(x)+ self.lin_h(h) ) 
    
    def forward(self,x):
        l = x.shape[0]
        batch = x.shape[1]
        h = torch.zeros(batch, self._dim_latent)
        sequence = [h] 
        for i in range(l) : 
            hi = self.one_step(x[i], sequence[-1])
            sequence.append(hi)
        return sequence[1:]

    def decode(self, h, type="many-to-one"):
        if type=="many-to-one":
            return self.lin_d(h)
        else : 
            seq = []
            for hi in h: 
                seq.append(self.lin_d(hi))
                
            return torch.stack(seq)
    
    def embedding(self,x):
        lin_e = torch.nn.Embedding(self._dim_input, self._dim_emb)
        return lin_e(x)


class LSTM(RNN):
    #  TODO:  Implémenter un LSTM
    def __init__(self, dim_input, dim_latent, dim_output, dim_emb):
        # x_emb : (length, batch, dim_emb)
        # h : (batch, dim_latent)
        # c : (batch, dim_latent)

        super().__init__(dim_input, dim_latent, dim_output, dim_emb)
        
        self._tanh = nn.Tanh() 
        self._sig = nn.Sigmoid()
        
        self.lin_ft = nn.Linear(dim_emb+dim_latent, dim_latent)
        self.lin_it = nn.Linear(dim_emb+dim_latent, dim_latent)
        self.lin_ct = nn.Linear(dim_emb+dim_latent, dim_latent)
        self.lin_ot = nn.Linear(dim_emb+dim_latent, dim_latent)

    def one_step(self, x_t, h, c):
        # x_t : (batch, dim_emb)
        # h_t-1 : (batch, dim_latent)
        # c_t-1 : (batch, dim_latent)
        
        concat = torch.cat((h,x_t), dim=1)
        ft = self._sig(self.lin_ft(concat))
        it = self._sig(self.lin_it(concat))
        ct = ft * c + it * self._tanh(self.lin_ct(concat))
        ot = self._sig(self.lin_ot(concat))
        ht = ot * self._tanh(ct)
        return ht, ct
        
    def forward(self, x):
        l = x.shape[0]
        batch = x.shape[1]
        h = torch.zeros(batch, self._dim_latent)
        c = torch.zeros(batch, self._dim_latent)
        sequence_h = [h] 
        sequence_c = [c]
        for i in range(l) : 
            hi, ci = self.one_step(x[i], sequence_h[-1], sequence_c[-1])
            sequence_h.append(hi)
            sequence_c.append(ci)

        return sequence_h[1:], sequence_c[1:]


class GRU(RNN):
    #  TODO:  Implémenter un LSTM
    def __init__(self, dim_input, dim_latent, dim_output, dim_emb):
        # x_emb : (length, batch, dim_emb)
        # h : (batch, dim_latent)
        # c : (batch, dim_latent)
        super().__init__(dim_input, dim_latent, dim_output, dim_emb)
        self._tanh = nn.Tanh() 
        self._sig = nn.Sigmoid()
        
        self.lin_zt = nn.Linear(dim_emb+dim_latent, dim_latent, bias = False)
        self.lin_rt = nn.Linear(dim_emb+dim_latent, dim_latent, bias = False)
        self.lin_h = nn.Linear(dim_emb+dim_latent, dim_latent, bias = False)

    def one_step(self, x_t, h):
        # x_t : (batch, dim_emb)
        # h_t-1 : (batch, dim_latent)
        # c_t-1 : (batch, dim_latent)
        
        concat = torch.cat((h,x_t), dim=1)
        zt = self._sig(self.lin_zt(concat))
        rt = self._sig(self.lin_rt(concat))

        concat2 = torch.cat((rt*h, x_t), dim=1)

        ht = (1-zt) * h + zt * self._tanh(self.lin_h(concat2))
      
        return ht



#  TODO:  Reprenez la boucle d'apprentissage, en utilisant des embeddings plutôt que du one-hot


PATH = "data/"
BATCH_SIZE = 32
dim_input = len(id2lettre)

def learn(modele, data, epoch = 10, dim_input = len(id2lettre), is_lstm = False, dim_latent = 50,dim_output = dim_input, dim_emb = 80, batch_size=BATCH_SIZE, path=PATH, ):

    writer = SummaryWriter("trump/"+datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))



    optimizer = torch.optim.Adam(modele.parameters(), lr=1e-2, weight_decay=1e-5)


    liste_loss = []
    liste_loss_test = []
    for epoch in tqdm(range(epoch)):
        
        liste_loss_tmp = []

        for x in data:
            y = torch.vstack((x[1:],  torch.ones(x.shape[1])*PAD_IX))

            x_emb = modele.embedding(x.long())
            optimizer.zero_grad()
            
            if is_lstm:
                liste_h,_ = modele(x_emb) 
            else : 
                liste_h = modele(x_emb) 

            yhat = modele.decode(liste_h, "many-to-many")

            loss = maskedCrossEntropy(yhat, y.long(), PAD_IX)


            writer.add_scalar("Loss/train", loss, epoch)
            loss.backward()

            # Utilisez TensorBoard pour surveiller la magnitude des gradients
            #for name, param in modele.named_parameters():
            #    if param.grad is not None:
            #        writer.add_histogram(name + '/grad', param.grad, epoch)

            # Implémentez le "gradient clipping"
            torch.nn.utils.clip_grad_norm_(modele.parameters(), max_norm=1.0)  
            
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


####################################################################
############################## RNN #################################
####################################################################


BATCH_SIZE = 32

PATH = "data/"

data_trump = DataLoader(TextDataset(open(PATH+"trump_full_speech.txt","rb").read().decode(),maxlen=100), collate_fn=pad_collate_fn, batch_size= BATCH_SIZE
, shuffle=True)


dim_input = len(id2lettre)
dim_latent = 50
dim_output = dim_input
dim_emb = 80

epoch = 10


rnn = RNN(dim_input, dim_latent, dim_input, dim_emb)
learn(rnn, data_trump)

s = generate(rnn, rnn.embedding, rnn.decode, EOS_IX, start="Hello", maxlen=200)
print(s)



####################################################################
############################# LSTM #################################
####################################################################


BATCH_SIZE = 32

PATH = "data/"

data_trump = DataLoader(TextDataset(open(PATH+"trump_full_speech.txt","rb").read().decode(),maxlen=100), collate_fn=pad_collate_fn, batch_size= BATCH_SIZE
, shuffle=True)


dim_input = len(id2lettre)
dim_latent = 50
dim_output = dim_input
dim_emb = 80

epoch = 10


lstm = LSTM(dim_input, dim_latent, dim_input, dim_emb)
learn(lstm, data_trump, is_lstm=True)

s = generate(lstm, lstm.embedding, lstm.decode, EOS_IX, start="Hello", maxlen=200,  is_lstm=True)
print(s)




####################################################################
############################## GRU #################################
####################################################################

BATCH_SIZE = 32

PATH = "data/"

data_trump = DataLoader(TextDataset(open(PATH+"trump_full_speech.txt","rb").read().decode(),maxlen=100), collate_fn=pad_collate_fn, batch_size= BATCH_SIZE
, shuffle=True)


dim_input = len(id2lettre)
dim_latent = 50
dim_output = dim_input
dim_emb = 80

epoch = 10


gru = GRU(dim_input, dim_latent, dim_input, dim_emb)
learn(gru, data_trump)

s = generate(gru, gru.embedding, gru.decode, EOS_IX, start="Hello ", maxlen=200)
print(s)



####################################################################
########################## beam search #############################
####################################################################


s = generate_beam(gru, gru.embedding, gru.decode, id2lettre[EOS_IX], 2, start="Hello", maxlen=10)
print(s)