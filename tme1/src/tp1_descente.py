import torch
from torch.utils.tensorboard import SummaryWriter
from tp1 import MSE, Linear, Context
import matplotlib.pyplot as plt


# Les données supervisées
x = torch.randn(50, 13)
y = torch.randn(50, 3)

# Les paramètres du modèle à optimiser
w = torch.randn(13, 3)
b = torch.randn(3)

epsilon = 0.05


writer = SummaryWriter()

linear = Linear()
mse = MSE()

liste_loss = []
for n_iter in range(100):
    loss_cxt=Context()
    linear_cxt=Context()

    ##  TODO:  Calcul du forward (loss)
    yhat = linear.forward(linear_cxt,x,w,b)
    loss = mse.forward(loss_cxt, yhat, y)
    liste_loss.append(loss)


    # `loss` doit correspondre au coût MSE calculé à cette itération
    # on peut visualiser avec
    # tensorboard --logdir runs/
    writer.add_scalar('Loss/train', loss, n_iter)

    # Sortie directe
    print(f"Itérations {n_iter}: loss {loss}")

    ##  TODO:  Calcul du backward (grad_w, grad_b)
    yhat_backward, _ = mse.backward(loss_cxt, 1)
    _, grad_w, grad_b = linear.backward(linear_cxt, yhat_backward)

    ##  TODO:  Mise à jour des paramètres du modèle

    w = w - epsilon*grad_w
    b = b - epsilon*grad_b

plt.figure()
plt.plot(torch.arange(100),liste_loss)
plt.title("Loss")
plt.xlabel("Iterations")
plt.ylabel("loss")
plt.show()