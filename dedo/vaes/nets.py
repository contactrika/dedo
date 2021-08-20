#
# NNs for VAEs.
#
# @contactrika
#
from collections import OrderedDict

import torch.nn as nn

from .prob import GaussianDiagDistr


class MlpGauss(nn.Module):
    """A simple MLP for encoder/decoder that collapses sequential info."""
    def __init__(self, inp_dim, seq_len, out_dim,
                 hidden_layers=(128, 128, 128), activation=nn.LeakyReLU):
        super(MlpGauss, self).__init__()
        assert(len(hidden_layers) > 0), 'Need non-empty hidden_layers list'
        net = OrderedDict()
        last_layer_size = inp_dim*seq_len
        for l, layer_size in enumerate(hidden_layers):
            net['fcon%d' % l] = nn.Linear(last_layer_size, layer_size)
            net['nl%d' % l] = activation()
            last_layer_size = layer_size
        self.net = nn.Sequential(net) if len(hidden_layers) > 0 else None
        self.mu = nn.Linear(last_layer_size, out_dim)
        # Diagonal covariance for now; logvar, so no softplus.
        self.logvar = nn.Linear(last_layer_size, out_dim)

    def forward(self, inp):
        if len(inp.shape) == 2:
            bsz, inp_dim = inp.shape
            seq_len = 1
        else:
            bsz, seq_len, inp_dim = inp.shape
        out = self.net(inp.view(bsz*seq_len, inp_dim))
        gauss = GaussianDiagDistr(self.mu(out), self.logvar(out))
        return gauss
