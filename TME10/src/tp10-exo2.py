import torch
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import seaborn as sns
from utils import *
sns.set()


#  TODO:  Générer les heatmap

d_model = 100  
max_len = 500
pos_enc = PositionalEncoding(d_model, max_len)

sequence = torch.zeros(1, 50, d_model)  

x_with_pos_enc = pos_enc(sequence)

positional_embeddings = pos_enc.pe.squeeze(0).detach().numpy()

# Similarité entre les embeddings de position
similarity = torch.mm(x_with_pos_enc.squeeze(0), x_with_pos_enc.squeeze(0).t()).detach().numpy()

plt.figure()
sns.heatmap(similarity, cmap='viridis')
plt.title("Similarité entre les embeddings positionnels")
plt.xlabel("Position")
plt.ylabel("Position")
plt.show()
