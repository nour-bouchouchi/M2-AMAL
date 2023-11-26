from textloader import  string2code, id2lettre, code2string
import math
import torch

#  TODO:  Ce fichier contient les différentes fonction de génération

def generate(rnn, emb, decoder, eos, start="", maxlen=200, is_lstm=False):
    """  Fonction de génération (l'embedding et le decodeur être des fonctions du rnn). Initialise le réseau avec start (ou à 0 si start est vide) et génère une séquence de longueur maximale 200 ou qui s'arrête quand eos est généré.
        * rnn : le réseau
        * emb : la couche d'embedding
        * decoder : le décodeur
        * eos : ID du token end of sequence
        * start : début de la phrase
        * maxlen : longueur maximale
    """
    #  TODO:  Implémentez la génération à partir du RNN, et d'une fonction decoder qui renvoie les logits (logarithme de probabilité à une constante près, i.e. ce qui vient avant le softmax) des différentes sorties possibles
    sentence = start
    if start=="":
        h_i = torch.zeros(1,rnn._dim_latent)
        c_i = torch.zeros(1,rnn._dim_latent)
    else : 
        start = string2code(start)
        start = emb(start).unsqueeze(1)
        h_i = rnn(start)[-1]
        
        if is_lstm : 
            h_i,c_i = rnn(start)
            h_i = h_i[-1]
            c_i = c_i[-1]
        else : 
            h_i = rnn(start)[-1]


    for _ in range(maxlen):
        yhat = decoder(h_i, type="many-to-one").softmax(dim=1)
        x_i = torch.multinomial(yhat, 1)[0]
        if x_i[0] == eos : 
            break
        
        car = code2string(x_i)
        sentence += car

        if is_lstm : 
            h_i,c_i = rnn.one_step(emb(x_i), h_i, c_i)
        else : 
            h_i = rnn.one_step(emb(x_i), h_i)

    return sentence


def generate_beam(rnn, emb, decoder, eos, k, start="", maxlen=200):
    """
        Génere une séquence en beam-search : à chaque itération, on explore pour chaque candidat les k symboles les plus probables; puis seuls les k meilleurs candidats de l'ensemble des séquences générées sont conservés (au sens de la vraisemblance) pour l'itération suivante.
        * rnn : le réseau
        * emb : la couche d'embedding
        * decoder : le décodeur
        * eos : ID du token end of sequence
        * k : le paramètre du beam-search
        * start : début de la phrase
        * maxlen : longueur maximale
    """
    #  TODO:  Implémentez le beam Search

    if start=="":
        h_i = torch.zeros(1,rnn._dim_latent)
    else : 
        start_code = string2code(start)
        start_emb = emb(start_code).unsqueeze(1)
        h_i = rnn(start_emb)[-1]

    K = [(start, h_i, 1)]
    for lg in range(maxlen):
        #print("---------- "+str(lg)+" --------------")
        k_temp = []
        for seq, h, p in K:
            
            h_t = h

            yhat = decoder(h_t, type="many-to-one").softmax(dim=1)
            top_k_probas, top_k_x_i = torch.topk(yhat, k)
            top_k_probas = [torch.log(p) for p in top_k_probas]
            liste_seq = code2string(top_k_x_i[0])
            k_temp += [(seq+liste_seq[i], h, p+top_k_probas[0][i]) for i in range(k)]

        k_temp.sort(key = lambda x: -x[2].item())
        k_temp = k_temp[:k]
        K = [(seq, rnn.one_step(emb(string2code(seq[-1])),h_i), p) if seq[-1]!=eos else (seq, h_i, p) for seq, h_i, p in k_temp ]
        k_temp = None
    K.sort(key = lambda x: -x[2])
    return K[0][0]


# p_nucleus
def p_nucleus(decoder, alpha: float):
    """Renvoie une fonction qui calcule la distribution de probabilité sur les sorties

    Args:
        * decoder: renvoie les logits étant donné l'état du RNN
        * alpha (float): masse de probabilité à couvrir
    """
    def compute(h):
        """Calcule la distribution de probabilité sur les sorties

        Args:
           * h (torch.Tensor): L'état à décoder
        """
        #  TODO:  Implémentez le Nucleus sampling ici (pour un état s)
    return compute
