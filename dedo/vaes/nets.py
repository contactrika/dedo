"""
Nets for SVAE variants.


Note: this code is for research i.e. quick experimentation; it has minimal
comments for now, but if we see further interest from the community -- we will
add further comments, unify the style, improve efficiency and add unittests.

@contactrika

"""
import torch
import torch.nn as nn

from . import prob


class ConvStack(nn.Module):
    IMAGE_SIZES = [64, 128, 256, 512]

    def __init__(self, pr):
        super(ConvStack, self).__init__()
        self.debug = pr.debug
        # Note: Conv2d(3, 64, 4, 2, 1) means num input channels/filters is 3,
        # num output channels/filters is 64. 3rd arg specifies that kernel size
        # (filter size) is 4x4, 4th arg=2 means the kernel slides over the image
        # in strides of 2. This implies the image size will be halved.
        # Last arg means add 1 padding to avoid shrinking output further.
        # Larger stride is used here instead of maxpool:
        # https://stackoverflow.com/questions/44666390/max-pool-layer-vs-convolution-with-stride-performance
        # Note: dropout sometimes helps conv too:
        # https://www.reddit.com/r/MachineLearning/comments/42nnpe/why_do_i_never_see_dropout_applied_in/
        # Dropout is crucial to ensure stability when using small datasets.
        # There is a debate as to whether apply BatchNorm before or after nl:
        # https://www.reddit.com/r/MachineLearning/comments/67gonq/d_batch_normalization_before_or_after_relu/
        # TCVAE code applies it before. Works just as well when applying after.
        # Conv net from:
        # https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html
        nflt = pr.conv_nfilters
        in_knlsz = 4  # inner kernel size should be always 4
        knlsz0 = 4 if pr.im_sz == 64 else 8
        strdsz0 = int(knlsz0/2)
        pdsz0 = int(strdsz0/2)
        in_strdsz = int(in_knlsz/2)
        in_pdsz = int(in_strdsz/2)
        assert(pr.im_sz in ConvStack.IMAGE_SIZES)
        self.conv = nn.Sequential(  # in: 3x 64x64, 128x128 or 256x256
            nn.Conv2d(3, nflt, knlsz0, strdsz0, pdsz0, bias=False),
            nn.BatchNorm2d(nflt), pr.nl,
            nn.Conv2d(nflt, nflt*2, in_knlsz, in_strdsz, in_pdsz, bias=False),
            nn.BatchNorm2d(nflt*2), pr.nl,
            nn.Conv2d(nflt*2, nflt*2, pr.knl_sz, pr.strd_sz, pr.pd_sz, bias=False),
            nn.BatchNorm2d(nflt*2), pr.nl)
        n_more = 1 if pr.im_sz == 256 else 2 if pr.im_sz == 512 else 0
        for i in range(n_more):
            self.conv.add_module(
                f'conv_more{i:d}', nn.Conv2d(
                    nflt*2, nflt*2, pr.knl_sz, pr.strd_sz, pr.pd_sz, bias=False))
            self.conv.add_module(f'conv_more{i:d}_nl', pr.nl)
            self.conv.add_module(f'conv_more{i:d}_bn', nn.BatchNorm2d(nflt*2))
        self.conv.add_module('conv_head', nn.Conv2d(
            nflt*2, nflt*4, in_knlsz, in_strdsz, in_pdsz, bias=False))
        self.conv.add_module('conv_head_bn', nn.BatchNorm2d(nflt*4))
        self.conv.add_module('conv_head_nl', pr.nl)
        self.conv.add_module('conv_out', nn.Conv2d(
            nflt*4, pr.comp_out_sz, in_knlsz, 1, 0, bias=False))
        self.conv.add_module('conv_out_nl', pr.nl)
        print('Constructed ConvStack', self)

    def forward(self, x_1toT):
        assert(x_1toT.dim() == 5)  # bsz, seq_len, clr_chn, w, h
        sz = x_1toT.size()
        out = self.conv(x_1toT.view(sz[0]*sz[1], *sz[2:]))
        out = out.view(sz[0], sz[1], -1)  # 1D output code
        return out


class ConvDecoder(nn.Module):
    """
    Applies de-conv to generate x_t.
    Together with priors, specifies the generative model:
    p(x_{1:T}, z_{1:T}) = prod_{t=1}^T p(z_t | z_{t-1}) p(x_t | z_t).
    """
    def __init__(self, pr, in_sz):
        super(ConvDecoder, self).__init__()
        self.debug = pr.debug
        nflt = pr.conv_nfilters
        # https://discuss.pytorch.org/t/pytorch-equivalent-of-tensorflow-conv2d-transpose-filter-tensor/16853/3
        in_knlsz = 4  # inner kernel size should be always 4
        in_strdsz = int(in_knlsz/2)
        in_pdsz = int(in_strdsz/2)
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(in_sz, nflt*4, in_knlsz, 1, 0, bias=False),
            pr.nl, nn.BatchNorm2d(nflt*4),
            nn.ConvTranspose2d(nflt*4, nflt*2, in_knlsz, in_strdsz, in_pdsz, bias=False),
            pr.nl, nn.BatchNorm2d(nflt*2),
            nn.ConvTranspose2d(nflt*2, nflt*2, pr.knl_sz, pr.strd_sz, pr.pd_sz, bias=False),
            pr.nl, nn.BatchNorm2d(nflt*2)
        )
        assert(pr.im_sz in ConvStack.IMAGE_SIZES)
        n_more = 1 if pr.im_sz == 256 else 2 if pr.im_sz == 512 else 0
        for i in range(n_more):
            self.deconv.add_module(
                f'deconv_more{i:d}', nn.ConvTranspose2d(
                    nflt*2, nflt*2, pr.knl_sz, pr.strd_sz, pr.pd_sz, bias=False))
            self.deconv.add_module(f'deconv_more{i:d}_nl', pr.nl)
            self.deconv.add_module(f'deconv_more{i:d}_bn', nn.BatchNorm2d(nflt*2))
        self.deconv.add_module('deconv_head', nn.ConvTranspose2d(
            nflt*2, nflt, in_knlsz, in_strdsz, in_pdsz, bias=False))
        self.deconv.add_module('deconv_head_nl', pr.nl)
        # self.deconv.add_module('deconv_head_bn', nn.BatchNorm2d(nflt))
        knlsz0 = 4 if pr.im_sz==64 else 8
        strdsz0 = int(knlsz0/2); pdsz0 = int(strdsz0/2)
        self.deconv.add_module('deconv_end', nn.ConvTranspose2d(
            nflt, pr.clr_chn, knlsz0, strdsz0, pdsz0, bias=False))
        # End with Sigmoid for pixels and masks in [0,1]
        self.deconv.add_module('deconv_sigmoid', nn.Sigmoid())
        print('Constructed ConvDecoder', self)

    def forward(self, latents_1toT):
        assert(latents_1toT.dim() == 3)
        sz = latents_1toT.size()
        in_deconv = latents_1toT.view(sz[0]*sz[1], sz[2], 1, 1)
        out = self.deconv(in_deconv)
        px_sz = out.size()[1:]
        out = out.view(sz[0], sz[1], *px_sz)
        return out


class EncoderDynamic(nn.Module):
    """
    Yields q_(z_{1:L} | x_feat_{1:T}, a_{1:L}), L>T
    """
    def __init__(self, pr, latent_sz):
        super(EncoderDynamic, self).__init__()
        self.in_seq_len = pr.hist
        self.out_seq_len = pr.past+pr.pred
        self.mu_nl = pr.mu_nl
        in_sz = (pr.comp_out_sz+pr.act_sz)*pr.hist
        out_sz = latent_sz*self.out_seq_len
        self.nn = nn.Sequential(nn.Linear(in_sz, pr.hidden_sz*2), pr.nl,
                                nn.Linear(pr.hidden_sz*2, pr.hidden_sz), pr.nl)
        # Keep mu and logvar separate for easier range restrictions.
        self.mu = nn.Sequential(nn.Linear(pr.hidden_sz, out_sz))
        if self.mu_nl is not None: self.mu.add_module('mu_nl', pr.nl)
        self.logvar = nn.Sequential(nn.Linear(pr.hidden_sz, out_sz), pr.logvar_nl)
        print('Constructed EncoderDynamic', self)

    def forward(self, x_1toT_feats, act_1toT):
        batch_sz, hist_seq_len, data_sz = x_1toT_feats.size()
        assert(hist_seq_len == self.in_seq_len)   # just a sanity check
        assert(act_1toT.size(1) == self.in_seq_len)  # just a sanity check
        inp_lst = [x_1toT_feats, act_1toT]
        in_1toT = torch.cat(inp_lst, dim=-1)
        inp = in_1toT.view(batch_sz, -1)
        nn_out = self.nn(inp)
        mus = self.mu(nn_out)
        logvars = self.logvar(nn_out)
        z_distr = prob.GaussianDiagDistr(
            mus.view(batch_sz*self.out_seq_len, -1),
            logvars.view(batch_sz*self.out_seq_len, -1))
        z_smpls = z_distr.sample_(require_grad=True)
        if self.mu_nl is not None: z_smpls = self.mu_nl(z_smpls)  # bound latent
        z_smpls = z_smpls.view(batch_sz, self.out_seq_len, -1)
        return z_smpls, z_distr


class EncoderDynamicRNN(nn.Module):
    """
    Yields q_(z_{1:T} | x_{1:T})
    """
    def __init__(self, pr, latent_sz, nolstm=True):
        super(EncoderDynamicRNN, self).__init__()
        # batch_first is set to True, so that the input and output tensors
        # have sizes [batch_size, seq_len, data_size].
        # Size of LSTM input is: size of each conv(x_t).
        self.dynamic_sz = latent_sz
        self.num_lstm_dir = 2  # bidirectional
        lstm_input_size = pr.comp_out_sz+pr.act_sz
        hidden_sz = pr.hidden_sz  # using same hidden size for all nets
        if nolstm:  # GRU for faster training
            self.gru = nn.GRU(lstm_input_size, pr.hidden_sz)
            self.lin = nn.Sequential(nn.Linear(pr.hidden_sz, pr.hidden_sz), pr.nl)
        else:
            self.lstm = nn.LSTM(input_size=lstm_input_size, batch_first=True,
                                hidden_size=hidden_sz, num_layers=1,
                                bidirectional=(self.num_lstm_dir>1))
            rnn_input_size = self.num_lstm_dir*hidden_sz
            self.rnn = nn.RNN(input_size=rnn_input_size, batch_first=True,
                              hidden_size=hidden_sz, num_layers=1)
        rnn_input_size = self.num_lstm_dir*hidden_sz
        self.rnn = nn.RNN(input_size=rnn_input_size, batch_first=True,
                          hidden_size=hidden_sz, num_layers=1)
        # keep mu and logvar separate for easier logs (logvar -> no softplus)
        self.mu_nl = pr.mu_nl
        self.mu = nn.Sequential(nn.Linear(pr.hidden_sz, self.dynamic_sz))
        if self.mu_nl is not None:
            self.mu.add_module('mu_nl', pr.nl)
        self.logvar = nn.Sequential(
            nn.Linear(pr.hidden_sz, self.dynamic_sz), pr.logvar_nl)
        print('Constructed EncoderDynamic', self)

    def forward(self, x_1toT_feats, act_1toT, f_smpl=None):
        # x_all_features should have shape [bsz, seq_length, data_size]
        bsz, seq_len, data_size = x_1toT_feats.size()
        # We don't explicitly set initial LSTM state, it defaults to zeros.
        inp_lst = [x_1toT_feats, act_1toT]
        if f_smpl is not None:
            f_smpls_all = f_smpl.detach().unsqueeze(1).repeat(1, seq_len, 1)
            inp_lst.append(f_smpls_all)
        inp = torch.cat(inp_lst, dim=-1)
        if hasattr(self, 'lstm'):
            lstm_out, _ = self.lstm(inp)
            # lstm_out is automatically the right input to nn.RNN, since it's
            # (bsz, seq_length, hidden_sz*2) when batch_first=True
            # we don't explicitly set initial RNN state, it defaults to zeros.
            rnn_out, _ = self.rnn(lstm_out)
        else:
            gru_out, _ = self.gru(inp)
            rnn_out = self.lin(gru_out)
        # RNN outputs r_t at each timestep. Each of these is fed (separately)
        # into the MLP that parameterizes means and logvars.
        qz_mus_lst = []; qz_logvars_lst = []; z_smpls = []
        for t in range(seq_len):
            rnn_out_t = rnn_out[:,t,:].view(bsz,-1)
            mu_out = self.mu(rnn_out_t)
            logvar_out = self.logvar(rnn_out_t)
            qz_t_distr = prob.GaussianDiagDistr(mu_out, logvar_out)
            z_smpls.append(qz_t_distr.sample_(require_grad=True))
            qz_mus_lst.append(qz_t_distr.mu)
            qz_logvars_lst.append(qz_t_distr.logvar)
        mus = torch.cat(qz_mus_lst, dim=1).view(bsz*seq_len, -1)
        logvars = torch.cat(qz_logvars_lst, dim=1).view(bsz*seq_len, -1)
        z_distr = prob.GaussianDiagDistr(mus, logvars)
        z_smpls = torch.stack(z_smpls, dim=1)
        if self.mu_nl is not None: z_smpls = self.mu_nl(z_smpls)  # bound latent
        z_smpls = z_smpls.view(bsz, seq_len, -1)
        return z_smpls, z_distr
