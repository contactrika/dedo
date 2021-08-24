#
# Sequential Autoencoder (SVAE) and Disentanged Sequential Autoencoder (DSA)
#
# @contactrika
#
from collections import OrderedDict
import numpy as np

import torch
import torch.nn as nn

from . import svae_utils  # used dynamically
from . import nets_advanced as nets
from .svae_utils import extract_tgts
from .prob import get_log_lik, GaussianDiagDistr


class SVAE(nn.Module):
    def __init__(self, im_sz, act_sz, latent_sz, params_class, device):
        super(SVAE, self).__init__()
        pr = eval('svae_utils.' + params_class)
        pr.im_sz = im_sz
        pr.act_sz = act_sz
        pr.latent_sz = latent_sz
        if pr.dynamic_sz is None:
            pr.dynamic_sz = latent_sz*2  # low dim * 2
        self.pr = pr
        self.pr_name = params_class
        self.device = device
        self.conv_stack = nets.ConvStack(pr)
        latent_sz = pr.static_sz + pr.dynamic_sz  # for now use a single NN
        if pr.past+pr.pred==1 or pr.pred>0:
            self.encoder = nets.EncoderDynamic(pr, latent_sz)
        else:
            self.encoder = nets.EncoderDynamicRNN(
                pr, latent_sz, nolstm=True)
        self.decoder = nets.ConvDecoder(pr, latent_sz)
        self.to(self.device)
        print('Created SVAE w/ latent size', latent_sz, 'on', self.device)

    def latent_sz(self):
        return (self.pr.past+self.pr.pred)*(self.pr.static_sz+self.pr.dynamic_sz)

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


class DSA(nn.Module):
    def __init__(self, im_sz, act_sz, latent_sz, params_class, device):
        super(DSA, self).__init__()
        pr = eval('svae_params.'+params_class)
        pr.im_sz = im_sz
        pr.act_sz = act_sz
        if pr.static_sz is None:
            assert(pr.dynamic_sz is None)
            pr.static_sz = latent_sz
            pr.dynamic_sz = latent_sz
        if pr.dynamic_sz is None: pr.dynamic_sz = latent_sz*2  # low dim * 2
        self.pr = pr
        self.pr_name = params_class
        self.device = device
        self.conv_stack = nets.ConvStack(pr)
        self.encoder_static = nets.EncoderStatic(pr, nolstm=False)
        self.encoder_dynamic = nets.EncoderDynamicRNN(pr, nolstm=False)
        self.decoder = nets.ConvDecoder(pr, pr.static_sz + pr.dynamic_sz)
        self.static_prior = nets.LearnableGaussianDiagDistr(pr)
        self.dynamic_prior = nets.LearnableGaussianDiagCell(pr)
        self.to(self.device)
        print('Created DSA w/ latent size', pr.static_sz+pr.dynamic_sz,
              'on', self.device)

    def latent_sz(self):
        return self.pr.static_sz + (self.pr.past+self.pr.pred)*self.pr.dynamic_sz

    def latent_code(self, x_1toT, act_1toT, x_1toT_feats=None):
        bsz, seq_len, *_ = act_1toT.size()
        _, static_post_q, _, dyn_post_q = self.encode(
            x_1toT, act_1toT, x_1toT_feats)
        latent_code = [static_post_q.mu, dyn_post_q.mu.view(bsz, -1)]
        latent_code = torch.cat(latent_code, dim=-1).detach()
        return latent_code

    def recon(self, x_1toT, act_1toT, f_smpl=None, z_smpls=None, debug=False):
        assert ((type(x_1toT) == torch.Tensor) and (x_1toT.dim() == 5))
        batch_size, seq_len, clr_chnls, data_h, data_w = x_1toT.size()
        static_post_q = None; dyn_post_q = None
        if (f_smpl is None) or (z_smpls is None):
            assert((f_smpl is None) and (z_smpls is None))  # will replace both
            res = self.encode(x_1toT, act_1toT)
            f_smpl, static_post_q, z_smpls, dyn_post_q = res
        # Note that we run decoder for each x_t individually, not for the
        # whole sequence: z_smpls contain batch_size*seq_len entries.
        # We reuse f sampled for all timesteps in each batch.
        f_smpls = f_smpl.unsqueeze(1).repeat(1, seq_len, 1)
        latent_code = torch.cat([f_smpls, z_smpls], dim=-1)
        recon_xs = self.decoder(latent_code)
        return recon_xs, f_smpl, static_post_q, z_smpls, dyn_post_q

    def loss(self, x_1toL, act_1toL, kl_beta=1.0, debug=False):
        assert((type(x_1toL) == torch.Tensor) and (x_1toL.dim() == 5))
        batch_size, tot_seq_len, chnls, data_h, data_w = x_1toL.size()
        res = extract_tgts(x_1toL, act_1toL,
                           self.pr.hist, self.pr.past, self.pr.pred)
        x_1toT, act_1toT, _, xs_tgt, acts_tgt, _ = res
        res = self.recon(x_1toT, act_1toT)
        recon_xs, f_smpl, static_post_q, z_smpls, dyn_post_q = res
        recon_log_lik = get_log_lik(xs_tgt, recon_xs, lp=2)
        kl_static, kl_dynamic = self.compute_kl(
            act_1toT, f_smpl, static_post_q, z_smpls, dyn_post_q)
        elbo = recon_log_lik - (kl_static + kl_beta*kl_dynamic)
        debug_dict = {}
        if debug:
            debug_dict['recon_log_lik'] = recon_log_lik.mean().item()
            debug_dict['kl'] = (kl_static+kl_beta*kl_dynamic).mean().item()
            debug_dict['kl_static'] = kl_static.mean().item()
            debug_dict['kl_dynamic'] = kl_dynamic.mean().item()
        loss = elbo.mean().mul(-1)
        return loss, debug_dict

    def encode(self, x_1toT, act_1toT, x_1toT_feats=None):
        if x_1toT is not None:
            assert((type(x_1toT) == torch.Tensor) and (x_1toT.dim() == 5))
            x_1toT_feats = self.conv_stack(x_1toT)
        # Get static variational approximation posterior q(f|x_{1:T}).
        f_smpl, static_post_q = self.encoder_static(x_1toT_feats)
        # Construct variational approximation for the dynamic posterior
        # We get q(z_{1:T}|x_{1:T}) for all x_ts at once by viewing x_t
        # from different timesteps in the batch dimension.
        # The result is a diagonal Gaussian, since we assume independence for
        # all q(z_t|x_t) between all batch samples and timesteps (given x_t).
        z_smpls, dyn_post_q = self.encoder_dynamic(
            x_1toT_feats, act_1toT, f_smpl)
        return f_smpl, static_post_q, z_smpls, dyn_post_q

    def generate(self, batch_size, seq_len, act_1toT, f0=None, z0=None):
        # Note: we assume that act_1toT contain actions that will
        # precede each generated observation.
        x_1toT_list = []
        if f0 is not None:
            f_smpl = f0
        else:
            static_prior_distr = self.static_prior()
            f_smpl = static_prior_distr.sample_(
                require_grad=False, batch_size=batch_size)
        z_t, h_t, c_t = self.dynamic_prior.initial_state(batch_size)
        z_t_lst = []
        # Unlike during training, here we sample z_{t-1} from latent dynamics
        # (the so-called dynamic prior) p_{\theta}(z_t|z_{t-1}).
        # The encoders are not involved.
        for t in range(seq_len):
            pz_t_distr, h_t, c_t = self.dynamic_prior.forward(
                z_t, h_t, c_t, f_smpl, act_1toT[:,t])
            z_t = pz_t_distr.sample_(require_grad=False, batch_size=batch_size)
            if t==0 and z0 is not None: z_t = z0
            x = self.decoder.forward(f_smpl, z_t)
            x_1toT_list.append(x)
            z_t_lst.append(z_t)
        x_1toT = torch.stack(x_1toT_list, dim=1)
        return x_1toT

    def compute_kl(self, act_1toT, f_smpl, static_post_q, z_smpls, dyn_post_q):
        bsz, seq_len, _ = act_1toT.size()
        # Prior static part: log [ p(f) \prod_{t=1}^T p(z_t|z_{t-1}) ]
        static_prior_distr = self.static_prior()  # init Gaussian distr
        static_prior_log_prob = static_prior_distr.log_density_(f_smpl)
        # Prior dynamic part.
        pz_distr_lst, pz_joint_distr = self.run_latent_dynamics(
            z_smpls, f_smpl, act_1toT)
        logpz_list = []  # will eval E_{z~q} [log p(z_{1:T}|x)]
        for t in range(seq_len):
            logpz_list.append(pz_distr_lst[t].log_density_(z_smpls[:,t,:]))
        # sum over timesteps
        dynamic_prior_log_prob = torch.stack(logpz_list, dim=0).sum(0)
        # Posterior.
        # log [ q(f) \prod_{t=1}^T q(z_t|x_t) ] = log q(z_{1:T}|x_{1:T})
        static_posterior_log_prob = static_post_q.log_density_(f_smpl)
        dplp = dyn_post_q.log_density_(
            z_smpls.view(bsz*seq_len, -1)).view(bsz, seq_len, -1)
        dynamic_posterior_log_prob = dplp.sum(dim=1)  # sum  over t
        # Compute overall KL term.
        # KL(q(z|x)||p(z)) = E_q[log q(z|x)] - E_q[log p(z)]
        # Note: KL is additive for Gaussian with diag covar.
        analytic_kl = True
        if analytic_kl:
            kl_static = static_post_q.kl_to_other_distr_(static_prior_distr)
            kl_dynamic = dyn_post_q.kl_to_other_distr_(pz_joint_distr)
            kl_dynamic = kl_dynamic.view(bsz, seq_len, -1).sum(dim=1)  # sum t
        else:
            # Non-analytic KL could be better, since we might not want all q
            # regions to be close to p. We want this to 'hold' just for the
            # current q samples.
            kl_static = static_posterior_log_prob - static_prior_log_prob
            kl_dynamic = dynamic_posterior_log_prob - dynamic_prior_log_prob
        kl_dynamic = kl_dynamic.sum(dim=-1)  # leave only 0th dimension,
        kl_static = kl_static.sum(dim=-1)    # which is batch_size
        return kl_static, kl_dynamic

    def run_latent_dynamics(self, z_smpls, f_smpl, act_1toT):
        # Get results from latent_dynamics: compute log p(z_t|z_{t-1}) for each
        # timestep (this is needed to compute KL).
        batch_size, seq_len, _ = z_smpls.size()
        # We loop over timestaps explicitly.
        # This makes the code easy to understand and adapt.
        # Batching over time could be more efficient, but that would compromise
        # clarity. Clarity is crucial for research code, whereas performance
        # could be optimized later.
        pz_distr_lst = []; pz_mus_lst = []; pz_logvars_lst = []
        z_t, h_t, c_t = self.dynamic_prior.initial_state(batch_size)
        for t in range(seq_len):
            pz_t_distr, h_t, c_t = self.dynamic_prior.forward(
                z_t, h_t, c_t, f_smpl, act_1toT[:,t])
            pz_distr_lst.append(pz_t_distr)
            pz_mus_lst.append(pz_t_distr.mu)
            pz_logvars_lst.append(pz_t_distr.logvar)
            z_t = z_smpls[:,t,:]  # z_{t-1} from q sample for next step
        mus = torch.cat(pz_mus_lst, dim=1).view(batch_size*seq_len, -1)
        logvars = torch.cat(pz_logvars_lst, dim=1).view(batch_size*seq_len, -1)
        return pz_distr_lst, GaussianDiagDistr(mus, logvars)
