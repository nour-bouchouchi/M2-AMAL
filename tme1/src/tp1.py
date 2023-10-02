import torch
from torch.autograd import Function
from torch.autograd import gradcheck


class Context:
    """Un objet contexte très simplifié pour simuler PyTorch

    Un contexte différent doit être utilisé à chaque forward
    """
    def __init__(self):
        self._saved_tensors = ()
    def save_for_backward(self, *args):
        self._saved_tensors = args
    @property
    def saved_tensors(self):
        return self._saved_tensors


class MSE(Function):
    """Implementation de la fonction MSE"""
    @staticmethod
    def forward(ctx, yhat, y):
        ## Garde les valeurs nécessaires pour le backwards
        ctx.save_for_backward(yhat, y)

        #  TODO:  Renvoyer la valeur de la fonction
        return  torch.mean( torch.pow( yhat-y, 2) )

    @staticmethod
    def backward(ctx, grad_output):
        ## Calcul du gradient du module par rapport a chaque groupe d'entrées
        yhat, y = ctx.saved_tensors
        #  TODO:  Renvoyer par les deux dérivées partielles (par rapport à yhat et à y)
        return 2*(yhat-y)*grad_output / yhat.nelement(), 2*(y-yhat)*grad_output / y.nelement()

#  TODO:  Implémenter la fonction Linear(X, W, b)sur le même modèle que MSE

class Linear(Function):
    """Implementation de la fonction linéaire"""

    @staticmethod
    def forward(ctx, x, w, b):
        ctx.save_for_backward(x, w, b)
        return x @ w + b 

    @staticmethod
    def backward(ctx, grad_output):
        x,w,b = ctx.saved_tensors
        return grad_output  @ w.T , (grad_output.T @ x).T , torch.sum(grad_output, axis=0)
        

## Utile dans ce TP que pour le script tp1_gradcheck
mse = MSE.apply
linear = Linear.apply

