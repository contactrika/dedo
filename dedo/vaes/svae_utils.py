#
# NetsParams and utils for SVAE
#
# @contactrika
#
import torch


class SVAEParams():
    def __init__(self, hidden_size=512, static_size=8, dynamic_size=32,
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
        self.static_sz = static_size     # size of f in q(z_{1:T}, f | x_{1:T})
        self.dynamic_sz = dynamic_size   # size of z in q(z_{1:T}, f | x_{1:T})
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

#                                   hid    st  dyn  hist past pred
PARAMS_SVAE             = SVAEParams(512,  0,  None,   8,  8,  0)
PARAMS_PRED             = SVAEParams(512,  0,  None,   8,  8,  8)
PARAMS_DSA              = SVAEParams(512, None, None,  8,  8,  0)


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


def do_logging(epoch, debug_dict, debug_hist_dict, tb_writer):
    dbg_str = 'Train epoch {:d}'.format(epoch)
    if 'recon_log_lik' in debug_dict.keys():
        dbg_str += ' recon_log_lik: {:.4f}'.format(debug_dict['recon_log_lik'])
    print(dbg_str)
    if tb_writer is not None:
        for k,v in debug_dict.items():
            vv = v.mean().item() if type(v)==torch.Tensor else v
            tb_writer.add_scalar(k, vv, epoch)
        for k,v in debug_hist_dict.items():
            tb_writer.add_histogram(
                k,v.clone().cpu().data.numpy(), epoch)
