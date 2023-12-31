
import math
import click
from torch.utils.tensorboard import SummaryWriter
import logging
import re
from pathlib import Path
from tqdm import tqdm
import numpy as np
import time
from datamaestro import prepare_dataset
import torch.nn.functional as F
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchmetrics.classification import BinaryAccuracy
import matplotlib.pyplot as plt
from utils import *



MAX_LENGTH = 500

logging.basicConfig(level=logging.INFO)

class FolderText(Dataset):
    """Dataset basé sur des dossiers (un par classe) et fichiers"""

    def __init__(self, classes, folder: Path, tokenizer, load=False):
        self.tokenizer = tokenizer
        self.files = []
        self.filelabels = []
        self.labels = {}
        for ix, key in enumerate(classes):
            self.labels[key] = ix

        for label in classes:
            for file in (folder / label).glob("*.txt"):
                self.files.append(file.read_text() if load else file)
                self.filelabels.append(self.labels[label])

    def __len__(self):
        return len(self.filelabels)

    def __getitem__(self, ix):
        s = self.files[ix]
        return self.tokenizer(s if isinstance(s, str) else s.read_text()), self.filelabels[ix]
    def get_txt(self,ix):
        s = self.files[ix]
        return s if isinstance(s,str) else s.read_text(), self.filelabels[ix]

def get_imdb_data(embedding_size=50):
    """Renvoie l'ensemble des donnéees nécessaires pour l'apprentissage (embedding_size = [50,100,200,300])

    - dictionnaire word vers ID
    - embeddings (Glove)
    - DataSet (FolderText) train
    - DataSet (FolderText) test

    """
    WORDS = re.compile(r"\S+")

    words, embeddings = prepare_dataset(
        'edu.stanford.glove.6b.%d' % embedding_size).load()
    OOVID = len(words)
    words.append("__OOV__")
    word2id = {word: ix for ix, word in enumerate(words)}
    embeddings = np.vstack((embeddings, np.zeros(embedding_size)))

    def tokenizer(t):
        return [word2id.get(x, OOVID) for x in re.findall(WORDS, t.lower())]

    logging.info("Loading embeddings")

    logging.info("Get the IMDB dataset")
    ds = prepare_dataset("edu.stanford.aclimdb")

    return word2id, embeddings, FolderText(ds.train.classes, ds.train.path, tokenizer, load=False), FolderText(ds.test.classes, ds.test.path, tokenizer, load=False)

#  TODO: 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

###################################################  EXERCICE 1 ################################################### 
class Self_attention(nn.Module):
    def __init__(self, embed_dim, c=0):
        super().__init__()
        self.lin_k = nn.Linear(embed_dim, embed_dim, bias=False)
        self.lin_q = nn.Linear(embed_dim, embed_dim, bias=False)
        self.lin_v = nn.Linear(embed_dim, embed_dim, bias=False)
        self.soft = nn.Softmax(dim=1)
        self.lin = nn.Linear(embed_dim, embed_dim)
        self.c = torch.tensor(c).requires_grad_(False)
        self.relu = nn.ReLU()
        self._embed_dim = torch.tensor(embed_dim).requires_grad_(False)
    
    def softmax_masque(self, x, l):
        # faire un masque qui ne prenne pas en compte les tokens de padding
        # x: (batch_size, seq_len, emb_size)
        # l: (batch_size)

        mask = torch.arange(x.size(1))[None, :].to(device) >= l[:, None].to(device)
        #print('mask.shape',mask.shape)
       
        mask = mask.unsqueeze(2).expand(x.size())
        x[mask] = float('-inf')  #après le softmax cela tendera vers 0

        return self.soft(x)

    def forward(self, x, l):
        # x: (batch_size, seq_len_false, emb_size)
        # l: (seq_len_true)
        
        q = self.lin_q(x)
        #print('q.shape',q.shape)
        k = self.lin_k(x)
        #print('k.shape',k.shape)
        v = self.lin_v(x)
        #print('v.shape',v.shape)

        # softmax avec masque pour ne pas tenir compte du padding
        attention = self.softmax_masque(self.c + torch.div(q @ torch.transpose(k, dim0=1, dim1=2), torch.sqrt(self._embed_dim)), l)
        #print('attention.shape',attention.shape)
        
        x = self.lin( attention @ v )
        #print('x.shape',x.shape)
        x = self.relu(x)

        return x
    
            
class Model_attention(nn.Module):
    def __init__(self, embed_dim, L=3, c=0):
        super().__init__()
        self._embed_dim = embed_dim
        self.c = c

        self.att = nn.ModuleList(Self_attention(embed_dim, c) for _ in range(L))
        self.lin = nn.Linear(embed_dim, 1)
        self.sigmoid = nn.Sigmoid()       

    def forward(self, x, l):
        #print('x.shape', x.shape)
        for att_i in self.att : 
            x = att_i(x, l) 
        #print('x_att.shape',x.shape)
            
        x = torch.mean(x, dim=1)
        x = self.lin(x)
        #print('x_lin.shape',x.shape)
        
        x = self.sigmoid(x)
        #print('x_sig.shape',x.shape)

        return x

###################################################  EXERCICE 2 ################################################### 
class Self_attention_residuel(nn.Module):
    def __init__(self, embed_dim, c=0):
        super().__init__()
        self.lin_k = nn.Linear(embed_dim, embed_dim, bias=False)
        self.lin_q = nn.Linear(embed_dim, embed_dim, bias=False)
        self.lin_v = nn.Linear(embed_dim, embed_dim, bias=False)
        self.soft = nn.Softmax(dim=1)
        self.lin = nn.Linear(embed_dim, embed_dim)
        self.c = torch.tensor(c).requires_grad_(False)
        self.relu = nn.ReLU()
        self._embed_dim = torch.tensor(embed_dim).requires_grad_(False)
        self.norm = nn.LayerNorm(embed_dim)
        self.lin2 = nn.Linear(embed_dim, embed_dim)
    
    def softmax_masque(self, x, l):
        mask = torch.arange(x.size(1))[None, :].to(device) >= l[:, None].to(device)
        mask = mask.unsqueeze(2).expand(x.size())
        x[mask] = float('-inf')  
        return self.soft(x)

    def forward(self, x, l):
        x_norm = self.norm(x)  #normalise

        q = self.lin_q(x_norm)
        k = self.lin_k(x_norm)
        v = self.lin_v(x_norm)

        attention = self.softmax_masque(self.c + torch.div(q @ torch.transpose(k, dim0=1, dim1=2), torch.sqrt(self._embed_dim)), l)
        
        x_attention = self.lin( x + attention @ v )
        x_attention_relu = self.relu(x_attention)
        
        return x_attention_relu
    
            
class Model_residuel(nn.Module):
    def __init__(self, embed_dim, L=3, c=0):
        super().__init__()
        self._embed_dim = embed_dim
        self.c = c

        self.att = nn.ModuleList(Self_attention_residuel(embed_dim, c) for _ in range(L))
        self.lin = nn.Linear(embed_dim, 1)
        self.sigmoid = nn.Sigmoid()       

    def forward(self, x, l):
        for att_i in self.att : 
            x = att_i(x, l)             
        x = torch.mean(x, dim=1)
        x = self.lin(x)
        x = self.sigmoid(x)

        return x

###################################################  EXERCICE 3 ################################################### 

class Model_attention_positional_encoding(Model_attention):
    def __init__(self, embed_dim, L=3, c=0):
        super().__init__(embed_dim, L=L, c=c)
        self.pe = PositionalEncoding(embed_dim, MAX_LENGTH)      

    def forward(self, x, l):
        x = self.pe(x)

        for att_i in self.att : 
            x = att_i(x, l)             
        x = torch.mean(x, dim=1)
        x = self.lin(x)        
        x = self.sigmoid(x)
        return x

class Model_residuel_positional_encoding(Model_residuel):
    def __init__(self, embed_dim, L=3, c=0):
        super().__init__(embed_dim, L=L, c=c)
        self.pe = PositionalEncoding(embed_dim, MAX_LENGTH)      

    def forward(self, x, l):
        x = self.pe(x)
        for att_i in self.att : 
            x = att_i(x, l)             
        x = torch.mean(x, dim=1)
        x = self.lin(x)        
        x = self.sigmoid(x)
        return x


###################################################  EXERCICE 4 ################################################### 

class Model_attention_positional_encoding_cls(Model_attention):
    def __init__(self, embed_dim, L=3, c=0):
        super().__init__(embed_dim, L=L, c=c)
        self.pe = PositionalEncoding(embed_dim, MAX_LENGTH)  
        self.cls = torch.zeros(1,1,embed_dim).requires_grad_(True) 

    def forward(self, x, l):
        x = torch.cat([self.cls.expand(x.size(0), -1, -1), x], dim=1)
        x = self.pe(x)
        for att_i in self.att : 
            x = att_i(x, l)             
        x = torch.mean(x, dim=1)
        x = self.lin(x)        
        x = self.sigmoid(x)
        return x

class Model_residuel_positional_encoding_cls(Model_residuel):
    def __init__(self, embed_dim, L=3, c=0):
        super().__init__(embed_dim, L=L, c=c)
        self.pe = PositionalEncoding(embed_dim, MAX_LENGTH)      
        self.cls = torch.zeros(1,1,embed_dim).requires_grad_(True) 

    def forward(self, x, l):
        x = torch.cat([self.cls.expand(x.size(0), -1, -1), x], dim=1)
        x = self.pe(x)
        for att_i in self.att : 
            x = att_i(x, l)             
        x = torch.mean(x, dim=1)
        x = self.lin(x)        
        x = self.sigmoid(x)
        return x

###################################################  TRAINING ################################################### 
    

def eval(model, val_loader, device, criterion, writer, write_tensorboard, epoch):
    model.eval()

    acc_eval = BinaryAccuracy().to(device) 

    epoch_loss = []
    epoch_accuracy = []
    with torch.no_grad():
        for batch_idx, (data, target, l) in enumerate(val_loader):
            data, target, l = data.to(device), target.float().to(device), l.to(device)
            logits = model(data, l).view(-1).float()
            loss = criterion(logits, target)
            
            epoch_loss.append(loss.item())
            batch_accuracy = acc_eval(logits, target)
            epoch_accuracy.append(batch_accuracy.item())


        """
        if write_tensorboard:

            writer.add_scalar('Loss validation', np.mean(epoch_loss), epoch)
        
            for name, weight in model.named_parameters():
                writer.add_histogram(name, weight, epoch)
                writer.add_histogram(f'{name}.grad', weight.grad, epoch)

            entropie = criterion(logits,target)
            writer.add_histogram('Entropy validation', entropie, epoch)
            
            writer.add_scalar('Accuracy validation', np.mean(epoch_accuracy), epoch)
        """

        return np.mean(epoch_loss), np.mean(epoch_accuracy) 


def train(model, optimizer, train_loader, val_loader, nb_epoch, device, writer):
    criterion = nn.BCELoss().to(device) 
    acc_train = BinaryAccuracy().to(device) 
    loss_values = []
    accuracy_values = [] 
    loss_eval = []
    accuracy_values_eval = [] 
    write_tensorboard_train = nb_epoch//20

    for epoch in range(nb_epoch):
        print("epoch :",epoch)
        write_tensorboard = write_tensorboard_train != 0 and epoch % write_tensorboard_train == 0

        model.train()
        epoch_loss = []
        epoch_accuracy = []

        for i, (data, target, l) in enumerate(train_loader):
            data, target, l = data.to(device), target.float().to(device), l.to(device)
            optimizer.zero_grad()

            logits = model(data, l).view(-1).float()
            loss = criterion(logits, target)

            epoch_loss.append(loss.item())
            batch_accuracy = acc_train(logits, target)
            epoch_accuracy.append(batch_accuracy.item())

            loss.backward()
            optimizer.step()


        loss_values.append(np.mean(epoch_loss))
        accuracy_values.append(np.mean(epoch_accuracy))

        """
        if write_tensorboard:

            writer.add_scalar('Loss train', np.mean(epoch_loss), epoch)
        
            for name, weight in model.named_parameters():
                writer.add_histogram(f'{name}', weight, epoch)
                writer.add_histogram(f'{name}.grad', weight.grad, epoch)

            entropie = criterion(logits,target)
            writer.add_histogram('Entropy train', entropie, epoch)
            
            writer.add_scalar('Accuracy train', np.mean(epoch_accuracy), epoch)

        """

        epoch_loss_eval, epoch_acc_eval = eval(model, val_loader, device, criterion, writer, write_tensorboard, epoch)
        loss_eval.append(epoch_loss_eval)
        accuracy_values_eval.append(epoch_acc_eval)

        
    return loss_values, accuracy_values, loss_eval, accuracy_values_eval



@click.command()
@click.option('--test-iterations', default=1000, type=int, help='Number of training iterations (batches) before testing')
@click.option('--epochs', default=5, help='Number of epochs.')
@click.option('--modeltype', required=True, type=int, help="0: base, 1 : Attention1, 2: Attention2")
@click.option('--emb-size', default=100, help='embeddings size')
@click.option('--batch-size', default=32, help='batch size')
def main(epochs,test_iterations,modeltype,emb_size,batch_size):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    word2id, embeddings, train_data, test_data = get_imdb_data(emb_size)
    id2word = dict((v, k) for k, v in word2id.items())
    PAD = word2id["__OOV__"]
    embeddings = torch.Tensor(embeddings)
    emb_layer = nn.Embedding.from_pretrained(torch.Tensor(embeddings))

    def collate(batch):
        """ Collate function for DataLoader """
        data = [torch.LongTensor(item[0][:MAX_LENGTH]) for item in batch]
        lens = [len(d) for d in data]
        labels = [item[1] for item in batch]
        return emb_layer(torch.nn.utils.rnn.pad_sequence(data, batch_first=True,padding_value = PAD)).to(device), torch.LongTensor(labels).to(device), torch.Tensor(lens).to(device)


    train_loader = DataLoader(train_data, shuffle=True,batch_size=batch_size, collate_fn=collate)
    test_loader = DataLoader(test_data, batch_size=batch_size,collate_fn=collate,shuffle=False)

    ##  TODO: 
    ################### Exercice 1 ###################

    if modeltype==1 :
        model = Model_attention(emb_size, L=3, c=0).to(device)

    if modeltype==2 : #modèle résiduel (residual + layer norm)
        model = Model_residuel(emb_size, L=3, c=0).to(device)
    
    if modeltype==3 : #modèle positional embedding
        model = Model_attention_positional_encoding(emb_size, L=3, c=0).to(device)

    if modeltype==4 : # modèle residuel + positional embedding 
        model = Model_residuel_positional_encoding(emb_size, L=3, c=0).to(device)
    
    if modeltype==5 : # modèle positional embedding + cls
        model = Model_attention_positional_encoding_cls(emb_size, L=3, c=0).to(device)

    if modeltype==6 : # modèle residuel + positional embedding + cls
        model = Model_residuel_positional_encoding_cls(emb_size, L=3, c=0).to(device)

    lr=00.1
    optim = torch.optim.Adam(model.parameters(), lr=lr)
    writer = SummaryWriter()
    loss_values, accuracy_values, loss_eval, accuracy_values_eval = train(model, optim, train_loader, test_loader,epochs, device, writer)

    plt.figure()
    plt.plot(loss_values,label="Loss en train")
    plt.plot(loss_eval,label="Loss en validation")
    plt.legend()
    plt.show()


    plt.figure()
    plt.plot(accuracy_values,label="Accuracy en train")
    plt.plot(accuracy_values_eval,label="Accuracy en validation")
    plt.legend()
    plt.show()



if __name__ == "__main__":
    main()
    
