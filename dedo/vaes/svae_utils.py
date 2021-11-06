"""
Utils for SVAE.


Note: this code is for research i.e. quick experimentation; it has minimal
comments for now, but if we see further interest from the community -- we will
add further comments, unify the style, improve efficiency and add unittests.

@contactrika

"""
import torch


class SVAEParams():
    def __init__(self, hidden_size=512, dynamic_size=32,
                 hist=16, past=4, pred=8, logvar_limit=6,
                 mu_nl=torch.nn.Sigmoid(), conv_nflt=64, debug=False):
        self.clr_chn = 3                 # number of color channels (1 or 3)
        self.obj_sz = 28
        self.knl_sz = 4                    # conv kernel size
        self.strd_sz = int(self.knl_sz/2)  # conv stride size
        self.pd_sz = int(self.strd_sz/2)   # conv padding size
        self.conv_nfilters = conv_nflt     # number of conv filter
        self.comp_out_sz = 128           # size of inp stack output (e.g. conv)
        self.hidden_sz = hidden_size     # hidden layers for all nets
        self.dynamic_sz = dynamic_size   # size of z in q(z_{1:T} | x_{1:T})
        self.hist = hist
        self.past = past
        self.pred = pred
        assert(hist==0 or hist>=past)
        # ReLU does not have hyperparameters, works with dropout and batchnorm.
        # Other options like ELU/SELU are more suitable for very deep nets
        # and have shown some promise, but no huge gains.
        # With partialVAE ReLUs will cause variance to explode on high-dim
        # inputs like pixels from image.
        # Tanh can be useful when the range needs to be restricted,
        # but saturates and trains slower.
        # ELU showed better results for high learning rates on RL experiments.
        self.nl = torch.nn.ELU()
        # Control latent space range.
        self.mu_nl = mu_nl
        # Stabilize training by clamping logvar outputs.
        # sqrt(exp(-6)) ~= 0.05 so 6: std min=0.05 max=20.0
        # 10: std min=0.0067 max=148.4
        logvar_limit = logvar_limit
        self.logvar_nl = torch.nn.Hardtanh(-logvar_limit, logvar_limit)
        self.debug = debug

#                       hid    st   dyn  hist past pred
PARAMS_VAE  = SVAEParams(512,   0,   64,   1,  1,  0)
PARAMS_SVAE = SVAEParams(512,   0,   64,   4,  4,  0)
PARAMS_PRED = SVAEParams(512,   0,   64,   4,  4,  4)


def extract_tgts(x_1toL, act_1toL, hist, past, pred):
    assert(act_1toL.shape[1] >= past+pred), act_1toL.shape
    x_1toL = torch.unbind(x_1toL, dim=1)
    act_1toL = torch.unbind(act_1toL, dim=1)
    x_1toT = torch.stack(x_1toL[0:hist], dim=1)
    act_1toT = torch.stack(act_1toL[0:hist], dim=1)
    ofst = past+pred
    xs_tgt = torch.stack(x_1toL[-ofst:], dim=1)
    acts_tgt = torch.stack(act_1toL[-ofst:], dim=1)
    assert(xs_tgt.shape[1] == past+pred)
    assert(acts_tgt.shape[1] == past+pred)
    return x_1toT, act_1toT, xs_tgt, acts_tgt


def do_logging(epoch, debug_dict, debug_hist_dict, tb_writer, title_prefix):
    dbg_str = title_prefix+' epoch {:d}'.format(epoch)
    if 'recon_log_lik' in debug_dict.keys():
        dbg_str += ' recon_log_lik: {:.4f}'.format(debug_dict['recon_log_lik'])
    print(dbg_str)
    if tb_writer is not None:
        for k,v in debug_dict.items():
            vv = v.mean().item() if type(v)==torch.Tensor else v
            tb_writer.add_scalar(title_prefix+'_'+k, vv, epoch)
        for k,v in debug_hist_dict.items():
            tb_writer.add_histogram(
                title_prefix+'_'+k, v.clone().cpu().data.numpy(), epoch)


def fill_seq_bufs_from_rollouts(x_1toT, act_1toT, mask_1toT,
                                batch_size, seq_len, device):
    num_rlts = x_1toT.shape[0]
    rlt_len = x_1toT.shape[1]
    frames_1toL = torch.zeros(batch_size, seq_len, *x_1toT.shape[2:]).to(device)
    act_1toL = torch.zeros(batch_size, seq_len, *act_1toT.shape[2:]).to(device)
    for i in range(batch_size):
        bid = torch.randint(num_rlts, (1,))[0]
        tid = torch.randint(rlt_len-seq_len, (1,))[0]
        currbid_masks_1toL = mask_1toT[bid, tid:tid+seq_len].squeeze(-1)
        assert(len(currbid_masks_1toL) == seq_len)
        frames_1toL[i, :, :, :, :] = x_1toT[bid, tid:tid+seq_len, :, :, :]
        act_1toL[i, :, :] = act_1toT[bid, tid:tid+seq_len, :]
        # Mask should be 0 only at the start of the episode.
        # The offset works out to be one less than the id of the first
        # zero mask (one less because we want to replicate the frame just
        # before the 1st occurrence of mask==0).
        last_tid = None  # replicate last frame until the end
        if (currbid_masks_1toL[1:] < 1).any():
            tmp_done_2toL = torch.abs(1-currbid_masks_1toL[1:])
            res = tmp_done_2toL.nonzero(as_tuple=True)[0]
            next_episode_tid = 1 + res[0].item()
            last_tid = next_episode_tid - 1
        if last_tid is not None:
            frames_1toL[i, last_tid:, :] = x_1toT[bid, tid+last_tid, :]
    return frames_1toL.to(device), act_1toL.to(device)
