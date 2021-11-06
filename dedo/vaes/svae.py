"""
Various versions of sequential autoencoders.


Note: this code is for research i.e. quick experimentation; it has minimal
comments for now, but if we see further interest from the community -- we will
add further comments, unify the style, improve efficiency and add unittests.

@contactrika

"""
import torch
import torch.nn as nn

from . import svae_utils  # used dynamically
from . import nets
from .svae_utils import extract_tgts
from .prob import get_log_lik


class SVAE(nn.Module):
    def __init__(self, im_sz, act_sz, params_class, device):
        super(SVAE, self).__init__()
        pr = eval('svae_utils.' + params_class)
        pr.im_sz = im_sz
        pr.act_sz = act_sz
        self.pr = pr
        self.pr_name = params_class
        self.device = device
        self.conv_stack = nets.ConvStack(pr)
        latent_sz = pr.dynamic_sz  # for now use a single NN
        if pr.past+pr.pred == 1 or pr.pred > 0:
            self.encoder = nets.EncoderDynamic(pr, latent_sz)
        else:
            self.encoder = nets.EncoderDynamicRNN(
                pr, latent_sz, nolstm=True)
        self.decoder = nets.ConvDecoder(pr, latent_sz)
        self.to(self.device)
        print('Created SVAE w/ latent size', latent_sz, 'on', self.device)

    def latent_sz(self):
        return (self.pr.past + self.pr.pred) * self.pr.dynamic_sz

    def latent_code(self, x_1toT, act_1toT, x_1toT_feats=None):
        bsz = act_1toT.size(0)
        if x_1toT_feats is None: x_1toT_feats = self.conv_stack(x_1toT)
        z_smpls, z_distr = self.encoder(x_1toT_feats, act_1toT)
        return z_distr.mu.detach().view(bsz, -1)

    def recon(self, x_1toT, act_1toT):
        x_1toT_feats = self.conv_stack(x_1toT)
        z_smpls, z_distr = self.encoder(x_1toT_feats, act_1toT)
        recon_xs = self.decoder(z_smpls)
        return recon_xs, z_smpls, z_distr

    def loss(self, x_1toL, act_1toL, kl_beta=1.0, debug=False):
        assert((type(x_1toL) == torch.Tensor) and (x_1toL.dim() == 5))
        res = extract_tgts(x_1toL, act_1toL,
                           self.pr.hist, self.pr.past, self.pr.pred)
        x_1toT, act_1toT, xs_tgt, acts_tgt = res
        recon_xs, z_smpls, z_distr = self.recon(x_1toT, act_1toT)
        recon_log_lik = get_log_lik(xs_tgt, recon_xs, lp=2)
        loss = recon_log_lik.mean().mul(-1)
        kl = None
        debug_dict = {}
        if z_distr is not None and kl_beta>0:
            kl = z_distr.kl_to_standard_normal_().sum(dim=-1)
            loss = loss + kl_beta*kl.mean()
        if debug:
            debug_dict['recon_log_lik'] = recon_log_lik.mean().item()
            if kl is not None: debug_dict['kl'] = kl.mean().item()
        return loss, debug_dict
