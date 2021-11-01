#
# Nets for SVAE variants.
#
# @contactrika
#
from collections import OrderedDict
import torch
import torch.nn as nn

from ..vaes import prob


def decode_debug(unsup, x_1toT, act_1toT, f_smpl=None, z_smpls=None):
    with torch.no_grad():
        if hasattr(unsup, 'encoder_static'):  # DSA
            res = unsup.recon(x_1toT, act_1toT, f_smpl=f_smpl, z_smpls=z_smpls)
            recon_xs, f_smpl, _, z_smpls, _ = res
        else:  # VAE, SVAE
            f_smpl = None
            recon_xs, z_smpls, _ = unsup.recon(x_1toT, act_1toT)
        return recon_xs, f_smpl, z_smpls


def print_debug(msg, var_dict):
    print(msg, ' ', end='')
    for key, val in var_dict.items(): print(key, val.size())


def make_MLP(insz, outsz, hidden, nl, out_nl, drop=None):
    net = OrderedDict()
    prevsz = insz
    for l, h in enumerate(hidden):
        net['dense%d' % l] = nn.Linear(prevsz, h)
        net['nl%d' % l] = nl
        if drop is not None:
            net['drop%d' % l] = nn.Dropout(p=drop)
        prevsz = h
    net['out'] = nn.Linear(prevsz, outsz)
    if out_nl is not None: net['out_nl'] = out_nl
    return nn.Sequential(net)


class MlpStack4L(nn.Module):
    def __init__(self, clr_chn, im_sz, out_sz, hidden_sz, nl):
        super(MlpStack4L, self).__init__()
        hsz = hidden_sz
        self.nn = nn.Sequential(
            nn.utils.weight_norm(
                nn.Linear(clr_chn*im_sz*im_sz, hsz*4), name='weight'), nl,
            nn.utils.weight_norm(nn.Linear(hsz*4, hsz*2), name='weight'), nl,
            nn.utils.weight_norm(nn.Linear(hsz*2, hsz), name='weight'), nl,
            nn.Linear(hidden_sz, out_sz))
        print('Constructed MlpStack4L', clr_chn, im_sz, im_sz, '->', out_sz)

    def forward(self, x_1toT):
        sz = x_1toT.size()
        out = self.nn(x_1toT.view(sz[0]*sz[1], -1))
        return out.view(sz[0], sz[1], -1)


class MlpDecoder4L(nn.Module):
    def __init__(self, in_sz, clr_chn, im_sz, hidden_sz, nl):
        super(MlpDecoder4L, self).__init__()
        self.clr_chn = clr_chn
        self.im_sz = im_sz
        hz = hidden_sz
        self.nn = nn.Sequential(
            nn.utils.weight_norm(nn.Linear(in_sz, hz), name='weight'), nl,
            nn.utils.weight_norm(nn.Linear(hz, hz*2), name='weight'), nl,
            nn.utils.weight_norm(nn.Linear(hz*2, hz*4), name='weight'), nl,
            nn.Linear(hz*4, clr_chn*im_sz*im_sz), nn.Sigmoid())  # out in [0,1]
        print('Constructed MlpDecoder4L', in_sz, '->', clr_chn, im_sz, im_sz)

    def forward(self, latents_1toT):
        sz = latents_1toT.size()
        out = self.nn(latents_1toT.view(sz[0]*sz[1], -1))
        out = out.view(sz[0], sz[1], self.clr_chn, self.im_sz, self.im_sz)
        return out


#
# ------------------- DSA nets ----------------------------
#
class EncoderStatic(nn.Module):
    """
    EncoderStatic yields q(f | x_{1:T}).
    In combo with EncoderDynamic, it can be used to get:
    q(z_{1:T}, f | x_{1:T}) = q(f | x_{1:T}) q(z_{1:T} | f, x_{1:T})
    # Notes(rika): observe that the result of the sstatic encoder should be
    # invariant to permutations of x_ts within x_{1:T}.
    #
    """
    def __init__(self, pr, nolstm=True):
        super(EncoderStatic, self).__init__()
        insz = pr.comp_out_sz*(pr.past+pr.pred)//2  # will drop half the frames
        # batch_first is set to True, so that the input and output tensors
        # are provided as (batch, seq, feature).
        self.debug = pr.debug
        self.static_sz = pr.static_sz
        self.mu_nl = pr.mu_nl
        self.num_lstm_dir = 1  # note: the DSA paper did not use bidi here.
        if nolstm:
            # self.conv = ConvSeq2Seq(
            #     pr.comp_out_sz, pr.hist,
            #     pr.static_sz, pr.past+pr.pred, pr.hidden_sz,
            #     pr, n_conv_layers=2)
            self.nn = nn.Sequential(
                nn.Linear(insz, pr.hidden_sz*2), pr.nl,
                nn.Linear(pr.hidden_sz*2, pr.hidden_sz), pr.nl)
        else:
            self.lstm = nn.LSTM(
                input_size=pr.comp_out_sz, hidden_size=pr.hidden_sz,
                num_layers=1, batch_first=True,
                bidirectional=(self.num_lstm_dir>1))
        # keep mu and logvar separate for easier logs (logvar -> no softplus)
        # Same MLP as for the dynamic encoder.
        self.mu = nn.Sequential(nn.Linear(pr.hidden_sz*self.num_lstm_dir,
                                          self.static_sz), self.mu_nl)
        self.logvar = nn.Sequential(nn.Linear(
            pr.hidden_sz*self.num_lstm_dir, self.static_sz), pr.logvar_nl)
        print('Constructed EncoderStatic', self)

    def forward(self, x_1toT_feats):
        # x_all_features should be of size [batch_sz, seq_len, data_sz]
        assert (x_1toT_feats.dim() == 3)
        # A smart static encoder should not allow itself to see the whole seq.
        # Because it could encode features that influence the dynamics, but are
        # static for the whole seq - e.g. CoM. So use a random subset of frames.
        batch_sz, seq_len, data_sz = x_1toT_feats.size()
        perm = torch.randperm(seq_len)
        x_1toT_feats_in = x_1toT_feats[:,perm[0:int(max(1, seq_len/2))],:]
        if hasattr(self, 'lstm'):
            lstm_out, _ = self.lstm(x_1toT_feats_in)
            inp = lstm_out[:,-1,:]
        elif hasattr(self, 'conv'):
            inp = self.conv(x_1toT_feats_in)
        else:
            inp = self.nn(x_1toT_feats_in.view(batch_sz, -1))
        mu_out = self.mu(inp)
        logvar_out = self.logvar(inp)
        f_distr = prob.GaussianDiagDistr(mu_out, logvar_out)
        f_smpl = f_distr.sample_(require_grad=True)
        f_smpl = self.mu_nl(f_smpl)
        return f_smpl, f_distr


class LearnableGaussianDiagDistr(nn.Module):
    def __init__(self, pr):
        super(LearnableGaussianDiagDistr, self).__init__()
        # Do not use torch.augtograd.Variable to add custom params a Module:
        # https://stackoverflow.com/questions/51373919/
        # the-purpose-of-introducing-nn-parameter-in-pytorch
        self.mu = torch.nn.Parameter(torch.zeros(1,pr.static_sz))
        self.mu_nl = pr.mu_nl
        self.logvar = torch.nn.Parameter(torch.zeros(1,pr.static_sz))
        self.logvar_nl = pr.logvar_nl

    def forward(self):
        distr = prob.GaussianDiagDistr(
            self.mu_nl(self.mu), self.logvar_nl(self.logvar))
        return distr


class LearnableGaussianDiagCell(nn.Module):
    """
    p(z_{t+1}|z_t)
    Dynamics NN architecture.
    """
    def __init__(self,pr):
        super(LearnableGaussianDiagCell, self).__init__()
        self.debug = pr.debug
        self.dynamic_sz = pr.dynamic_sz
        self.hidden_sz = pr.hidden_sz
        self.nl = nn.ReLU()  # will be applied in forward()
        self.lstm_cell = nn.LSTMCell(
            input_size=pr.static_sz+pr.act_sz+self.dynamic_sz,
            hidden_size=self.hidden_sz)
        self.mlp = nn.Sequential(
            nn.Linear(self.hidden_sz, self.hidden_sz), self.nl)
        # keep mu and logvar separate for easier logs (logvar -> no softplus)
        self.mu = nn.Sequential(
            nn.Linear(self.hidden_sz, self.dynamic_sz), pr.mu_nl)
        self.logvar = nn.Sequential(
            nn.Linear(self.hidden_sz, self.dynamic_sz), pr.logvar_nl)
        # register_buffer is used when we want a stateful variable in the model
        # that is included in state_dict, but is not a parameter.
        # Below are our initial LSTM vals (nn.LSTM also uses zeros in h_0,c_0).
        self.register_buffer('z_0', torch.zeros(1,self.dynamic_sz))
        self.register_buffer('h_0', torch.zeros(1,self.hidden_sz))
        self.register_buffer('c_0', torch.zeros(1,self.hidden_sz))
        print('Constructed LearnableGaussianDiagCell', self)

    def initial_state(self, batch_sz):
        return self.z_0.expand(batch_sz, -1), \
               self.h_0.expand(batch_sz, -1), self.c_0.expand(batch_sz, -1)

    def forward(self, z_t, h_t, c_t, f_smpl, action):
        if self.debug: print_debug(
            'LearnableGaussianDiagCell',
            {'z_t':z_t, 'h_t':h_t, 'c_t':c_t, 'f_smpl':f_smpl, 'action':action})
        lstm_in = torch.cat([f_smpl, action, z_t], dim=1)
        h_tp1, c_tp1 = self.lstm_cell(lstm_in, (h_t, c_t))
        mlp_out = self.mlp(h_tp1)
        if self.debug: print_debug('LearnableGaussianDiagCell', {'mlp_out': mlp_out})
        mu_out = self.mu(mlp_out)
        logvar_out = self.logvar(mlp_out)
        if self.debug: print_debug('LearnableGaussianDiagCell',
                                   {'mu_out': mu_out, 'logvar_out': logvar_out})
        # Note: gradients will be propagated through mu, logvar.
        z_tp1_distr = prob.GaussianDiagDistr(mu_out, logvar_out)
        return z_tp1_distr, h_tp1, c_tp1
