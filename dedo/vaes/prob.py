#
# Utilities related to probability theory math.
# Could use libraries, but then would need to keep track of APIs.
# So coded the most needed parts here from scratch.
# Adapted from
# https://github.com/contactrika/bo-svae-dc/blob/master/svae-dc/svae_dc/utils/prob.py
#
# @contactrika
#
import math
import torch

LOGVAR_LIMIT = 10  # STD = sqrt(exp(-10)) = 0.0067  # avoid variance collapse


def get_log_lik(tgt_xs, recon_xs, lp=2):
    batch_size, seq_len, clr_chnls, data_h, data_w = tgt_xs.size()
    # log ( \prod_{t=1}^T p(x_t|z_t) )
    # We omit the constants log(2*pi) and log(|Sigma|), these are 'const',
    # since recon_x includes only the 'mean', var not modelled.
    log_lik = -0.5*torch.abs(tgt_xs - recon_xs)**lp
    log_lik = log_lik.sum(dim=[2,3,4])  # sum px
    log_lik = log_lik.mean(dim=1)   # mean over t
    return log_lik


class GaussianDiagDistr(object):
    """
    Multivariate Gaussian with diagonal covariance.
    Sigma_{ii} = exp(logvar[i])
    i.e. logvar holds logs of diagonal entries of the covariance matrix.
    A bit similar to using Independent,Normal from torch.distributions:
    Independent(Normal(loc, scale), 1).
    But this class offers finer control on logvar representation.
    Also uses much more explicit naming for class and the methods. So:
    Independent(Normal) -> LearnableGaussianDiagDistr
    sample() -> sample_no_grad()
    rsample() -> sample_with_grad()
    log_prob() -> log_density()
    and provides kl_from_sandard_normal()
    """
    def __init__(self, mu, logvar, logvar_limit=LOGVAR_LIMIT):
        super(GaussianDiagDistr, self).__init__()
        self.mu = mu
        # Sigma_{ii} = exp(logvar[i])
        # i.e. logvar holds logs of diagonal entries of the covariance matrix
        nl = torch.nn.Hardtanh(-logvar_limit, logvar_limit)
        self.logvar = nl(logvar)
        # Check that mu and logvar (if given as input) were as expected.
        GaussianDiagDistr.check_param_tensors(mu, logvar)

    @staticmethod
    def check_param_tensors(mu, logvar, debug=False):
        assert(mu.dim() == logvar.dim() == 2)  # [batch_size x Gaussian_dim]
        max_abs_logvar = 2*LOGVAR_LIMIT
        errs = (torch.abs(logvar)>max_abs_logvar).nonzero()
        if errs.size(0) > 0 and debug:
            print('logvar too extreme')
            for er in errs:
                msg = 'batch id %d dim %d logvar %0.4f'
                print(msg % (er[0], er[1], logvar[er[0], er[1]]))
            assert(False)

    @staticmethod
    def sample_no_grad(mu, logvar):
        # does not propagate gradients
        # similar to .sample() from torch.distributions
        with torch.no_grad():
            return GaussianDiagDistr.sample_with_grad(mu, logvar)

    @staticmethod
    def sample_with_grad(mu, logvar):
        # propagates gradients (i.e. uses reparameterization trick)
        # similar to .rsample() from torch.distributions
        std = torch.exp(0.5*logvar)
        eps = torch.empty_like(std).normal_()  # N(0,1) with same size as std
        return mu + eps*std

    @staticmethod
    def log_density(x, mu, logvar, omit, adjust=None, debug=False):
        assert(x.dim() == mu.dim() == logvar.dim())  # mu,logvar: [dim]
        batch_size, dimension = x.size()             # x: [bsz, dim]
        if mu.size(1) != dimension or logvar.size(1) != dimension:
            print('x', x.size(), 'vs mu', mu.size(), 'logvar', logvar.size())
            assert(False)
        # k-dim Gaussian with mu_{i} = mu[i], Sigma_{ii} = exp(logvar[i]).
        # Let inv_diag = Sigma^{-1}, core = (x-mu)^T*Sigma^{-1}*(x-mu); Compute:
        # ln N(x| mu, logvar) = -0.5[ln det(Sigma) - core - k*ln(2pi)]
        inv_diag = torch.exp(-1.0*logvar)
        delta = (x-mu)
        if debug:
            print('orig delta', delta)
            print('std', torch.exp(logvar).sqrt())
        if omit is not None:
            delta = torch.where(omit>=1, torch.zeros_like(delta), delta)
        if adjust is not None:
            delta = torch.mul(delta, adjust)
        # element-wise multiplication, we assume 0th dim is batch size.
        core = torch.mul(delta, inv_diag).mul(delta).sum(-1).view(-1, 1)
        log_det_sigma = logvar.sum(1).view(-1, 1)  # sum along non-batch dim
        c = dimension*math.log(2*math.pi)  # normalization constant
        return -0.5*(core + log_det_sigma + c)

    @staticmethod
    def kl_to_other(mu, logvar, other_mu, other_logvar, debug=False):
        # KL(q||p), with q,p being Gaussians with diagonal covariance.
        # We assume 0th dim is batch size.
        dim = mu.size(1); assert(dim == other_mu.size(1))
        other_inv_diag = torch.exp(-1.0*other_logvar)
        delta = (other_mu - mu)
        # element-wise multiplication, we assume 0th dim is batch size.
        core = torch.mul(delta, other_inv_diag).mul(delta).sum(-1).view(-1, 1)
        sigma_diag = torch.exp(logvar)
        tr = other_inv_diag.mul(sigma_diag).sum(-1).view(-1, 1)
        log_det_sigma = logvar.sum(1).view(-1, 1)  # sum along non-batch dim
        log_det_other_sigma = other_logvar.sum(1).view(-1, 1)
        last_term = log_det_other_sigma - log_det_sigma
        kl = 0.5*(tr + core - dim + last_term)
        if debug:
            print('max tr %s core %s last_term %s',
                  str(torch.max(tr).item()), str(torch.max(core)),
                  str(torch.max(last_term)))
            print('max mu %s other_mu %s', str(torch.max(mu)),
                         str(torch.max(other_mu)))
            print('min other_inv_diag %s', str(torch.min(other_inv_diag)))
            print('max other_inv_diag %s', str(torch.max(other_inv_diag)))
        tolerance = -1e-2
        if (kl<tolerance).any():
            print('ERROR: significantly negative KL ')
            print(torch.min(kl))
            assert(False)
        kl[kl<tolerance] = 0  # remove numerical instability side-effects if any
        return kl

    @staticmethod
    def kl_to_standard_normal(mu, logvar):
        # KL(q||p), with p being standard Normal.
        # From Appendix B of VAE "Auto-Encoding Variational Bayes", ICLR2014.
        # https://arxiv.org/abs/1312.6114
        # -KL(q||p) = 0.5 * sum(1 + log(sigma^2) - mean^2 - sigma^2)
        kl = 1 + 2*logvar - torch.pow(mu, 2) - torch.exp(logvar).pow(2)
        kl.mul_(-0.5)
        return kl

    def check_params_(self, debug=False):
        GaussianDiagDistr.check_param_tensors(self.mu, self.logvar, debug=debug)

    def sample_(self, require_grad=False):
        if require_grad:
            return GaussianDiagDistr.sample_with_grad(self.mu, self.logvar)
        else:
            return GaussianDiagDistr.sample_no_grad(self.mu, self.logvar)

    def log_density_(self, x, omit=None, adjust=None, debug=False):
        return GaussianDiagDistr.log_density(
            x, self.mu, self.logvar, omit=omit, adjust=adjust, debug=debug)

    def kl_to_other_mu_logvar_(self, other_mu, other_logvar):
        assert(type(other_mu) == type(other_logvar) == torch.Tensor)
        return GaussianDiagDistr.kl_to_other(
            self.mu, self.logvar, other_mu, other_logvar)

    def kl_to_other_distr_(self, other_distr, fixed_logvar=None):
        assert(type(other_distr) == GaussianDiagDistr)
        if fixed_logvar is not None:
            return GaussianDiagDistr.kl_to_other(
                self.mu, fixed_logvar, other_distr.mu, fixed_logvar)
        else:
            return GaussianDiagDistr.kl_to_other(
                self.mu, self.logvar, other_distr.mu, other_distr.logvar)

    def kl_to_standard_normal_(self):
        return GaussianDiagDistr.kl_to_standard_normal(self.mu, self.logvar)
