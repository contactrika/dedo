"""
Testing utilities.

python -m dedo.vaes.test_vae

@contactrika
"""
import matplotlib
# matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

import numpy as np

import torch

from dedo.vaes.svae import SVAE
from dedo.vaes.datasets import DeformEnvDataset, worker_init_fn
from dedo.utils.args import get_args


def plot_recon(svae, train_data, test_data, fig):
    fig.clear()
    n_rows, n_cols = 4, 2
    for row in range(n_rows):
        data = train_data if row < 2 else test_data

        x = next(data)
        inp_dim = np.prod(x.shape[:])
        x_hat_distr, _ = svae.reconstruct(x.view(1, inp_dim))
        x_hat = x_hat_distr.sample_().view(x.shape)
        plt.axis('off')
        fig.add_subplot(n_rows, n_cols, 1+row*n_cols)
        plt.title('train x' if row < 2 else 'test x')
        plt.imshow(x.squeeze(), cmap='gray')
        fig.add_subplot(n_rows, n_cols, 1+row*n_cols+1)
        plt.title('x_hat')
        plt.imshow(x_hat.squeeze(), cmap='gray')
    plt.ion()  # non-blocking
    plt.pause(0.05)
    plt.tight_layout()
    plt.show()


def test_vae(args):
    train_dataset = DeformEnvDataset(args)
    # test_dataset = DeformEnvDataset(args)
    # x = next(train_dataset)
    inp_dim = np.prod(args.cam_resolution**2 *3)
    svae = SVAE(inp_dim=inp_dim, inp_seq_len=1, out_dim=inp_dim)
    train_data_loader = torch.utils.data.DataLoader(
        train_dataset, num_workers=8, batch_size=24, shuffle=False, worker_init_fn=worker_init_fn)
    optim = torch.optim.Adam(svae.parameters(), lr=1e-3)
    n_epochs = 100
    fig = plt.figure(figsize=(6, 6))
    plot_loss_lst = []
    for epoch in range(n_epochs):
        # plot_recon(svae, train_dataset, train_dataset, fig)
        plot_loss_accum = 0
        for x in train_data_loader:
            show(x[0])
            show(x[1])
            show(x[5])
            show(x[8])
            show(x[9])
            show(x[10])
            show(x[11])
            x = x.view(x.shape[0], 1, inp_dim)
            recon_tgt = x.view(x.shape[0], inp_dim)
            optim.zero_grad()
            loss = -1.0*svae.elbo(x, recon_tgt)
            loss.backward()
            optim.step()
            plot_loss_accum += loss.item()
        plot_loss_lst.append(plot_loss_accum/len(train_dataset)/inp_dim)
        print(f'Epoch {epoch:d} loss {plot_loss_lst[-1]:0.4f}')
    fig.close()
def show(x):
    plt.imshow(x)
    plt.show()

if __name__ == '__main__':
    test_vae(get_args())
