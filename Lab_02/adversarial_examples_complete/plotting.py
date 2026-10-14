import argparse

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import LightSource
from torch import nn

import dataset
from neural_network import NeuralNetwork


def plot_images(X, y, yp, M, N, grayscale=False):
    f, ax = plt.subplots(M, N, sharex=True, sharey=True, figsize=(N, M*1.3))
    for i in range(M):
        for j in range(N):
            sample = 1 - X[i * N + j].cpu().numpy() if grayscale else X[i * N + j].cpu().numpy()
            ax[i][j].imshow(sample, cmap='gray' if grayscale else None)
            prediction = yp[i*N+j].item()
            title = ax[i][j].set_title(f'Pred: {prediction}')
            plt.setp(title, color=('g' if prediction == y[i*N+j] else 'r'))
            ax[i][j].set_axis_off()
    plt.tight_layout()
    plt.show()


def draw_loss(model, X, y, epsilon):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    Xi, Yi = np.meshgrid(np.linspace(-epsilon, epsilon, 100), np.linspace(-epsilon, epsilon, 100))

    def grad_at_delta(delta):
        delta.requires_grad_(True)
        nn.CrossEntropyLoss()(model(X + delta), y[0:1]).backward()
        return delta.grad.detach().sign().view(-1).cpu().numpy()

    dir1 = grad_at_delta(torch.zeros_like(X, requires_grad=True))
    delta2 = torch.zeros_like(X, requires_grad=True)
    delta2.data = torch.tensor(dir1).view_as(X).to(device)
    dir2 = grad_at_delta(delta2)
    np.random.seed(0)
    dir2 = np.sign(np.random.randn(dir1.shape[0]))

    all_deltas = torch.tensor((np.array([Xi.flatten(), Yi.flatten()]).T @
                               np.array([dir2, dir1])).astype(np.float32)).to(device)
    yp = model(all_deltas + X)
    Zi = nn.CrossEntropyLoss(reduction="none")(yp, y[0:1].repeat(yp.shape[0])).detach().cpu().numpy()
    Zi = Zi.reshape(*Xi.shape)
    # Zi = (Zi-Zi.min())/(Zi.max() - Zi.min())

    fig = plt.figure(figsize=(10, 10))
    ax = fig.gca(projection='3d')
    ls = LightSource(azdeg=0, altdeg=200)
    rgb = ls.shade(Zi, plt.cm.coolwarm)

    surf = ax.plot_surface(Xi, Yi, Zi, rstride=1, cstride=1, linewidth=0,
                           antialiased=True, facecolors=rgb)
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-file_name', help='model file name to load')
    args = parser.parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dataset_name = args.file_name.split('_')[1]
    data = dataset.get_dataset(dataset_name=dataset_name, device=device, bs=256)
    model = NeuralNetwork.load(args.file_name)
    model.to(device)
    x, y = data.trainloader.dataset.tensors[0][0:1].to(device), data.trainloader.dataset.tensors[1][0:1].to(device)
    draw_loss(model, x, y, 0.1)
