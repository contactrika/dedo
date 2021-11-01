#
# Sequential Variational Autoencoder (VAE).
#
# @contactrika
#
import torch
import torch.nn as nn

from .nets import MlpGauss


class SVAE(nn.Module):
    def __init__(self, inp_dim, inp_seq_len, out_dim, latent_dim=4):
        super(SVAE, self).__init__()
        self.encoder = MlpGauss(inp_dim, inp_seq_len, latent_dim)
        self.decoder = MlpGauss(latent_dim, 1, out_dim)

    def encode(self, x_1toT, require_grad=False):
        assert((type(x_1toT) == torch.Tensor) and (x_1toT.dim() == 3))
        latent_distr = self.encoder(x_1toT)  # q(z | x_{1:T})
        z_smpl = latent_distr.sample_(require_grad)
        return z_smpl, latent_distr

    def reconstruct(self, x_1toT, require_grad=False):
        assert ((type(x_1toT) == torch.Tensor) and (x_1toT.dim() == 3))
        z_smpl, latent_distr = self.encode(x_1toT, require_grad)
        recon_distr = self.decoder(z_smpl)
        return recon_distr, latent_distr

    def elbo(self, xi_1toT, recon_tgt):
        assert ((type(xi_1toT) == torch.Tensor) and (xi_1toT.dim() == 3))
        z_smpl, latent_distr = self.encode(xi_1toT, require_grad=True)
        # Reconstruction loss.
        recon_distr = self.decoder(z_smpl)
        ll_recon = recon_distr.log_density_(recon_tgt)
        # Latent KL loss.
        latent_kl = latent_distr.kl_to_sandard_normal_()
        latent_kl = latent_kl.sum(dim=1)  # sum over dims, ok since diag
        # Compute ELBO = recon - latent_kl
        # Multiply ELBO by -1 in the training (later) to make a loss.
        elbo = ll_recon - latent_kl
        return elbo.sum()
