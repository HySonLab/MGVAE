import torch 
import torch.nn as nn
from models.equiv_layers import *
from utils.graph_utils import *


class Decoder(nn.Module):
    def __init__(self, in_channel, out_edge_channel, out_node_channel, device):
        super().__init__()

        self.promote_1_to_2 = nn.Sequential(
            layer_1_to_2(in_channel, in_channel, device = device),
            layer_2_to_2(in_channel, in_channel, device = device)
        )
        self.node_predictor = nn.Sequential(nn.Linear(in_channel, in_channel * 2), nn.Tanh(), nn.Linear(in_channel * 2, out_node_channel))
        self.edge_predictor = nn.Sequential(nn.Linear(in_channel, in_channel * 2), nn.Tanh(), nn.Linear(in_channel * 2, out_edge_channel))
        self.out_edge_channel = out_edge_channel
    
    def forward(self, latent, mask = None):
        x = latent.transpose(1, 2)      
        node = self.node_predictor(x)
        latent = self.promote_1_to_2(latent)  
        edge = self.edge_predictor(latent.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        return node, edge
        
def KL_Gaussians(mu_encoder, L_encoder, mu_prior, L_prior, device):
    mu_0 = mu_encoder.transpose(1, 2)
    L_0 = L_encoder.transpose(2, 3).transpose(1, 2)
    sigma_0 = torch.matmul(L_0, L_0.transpose(2, 3))
    # sigma_0 = torch.matmul(L_0.transpose(2, 3), L_0)

    mu_1 = mu_prior.transpose(1, 2)
    L_1 = L_prior.transpose(2, 3).transpose(1, 2)
    sigma_1 = torch.matmul(L_1, L_1.transpose(2, 3))
    # sigma_1 = torch.matmul(L_1.transpose(2, 3), L_1)

    # Adding noise
    batch_size = mu_encoder.size(0)
    num_nodes = mu_encoder.size(1)
    num_channels = mu_encoder.size(2)

    noise = torch.cat([torch.eye(num_nodes).unsqueeze(dim = 0) for b in range(batch_size)]) * 1e-4
    noise = torch.cat([noise.unsqueeze(dim = 1) for c in range(num_channels)], dim = 1).to(device = device)

    sigma_0 += noise
    sigma_1 += noise

    sigma_1_inverse = torch.inverse(sigma_1)

    A = torch.matmul(sigma_1_inverse, sigma_0)
    A = torch.einsum('bcii->bc', A)

    x = mu_1 - mu_0
    B = torch.einsum('bci,bcij->bcj', x, sigma_1_inverse)
    B = torch.einsum('bcj,bcj->bc', B, x)

    sign_0, logabsdet_0 = torch.slogdet(sigma_0)
    sign_1, logabsdet_1 = torch.slogdet(sigma_1)

    logabsdet_0 = torch.where(torch.isnan(logabsdet_0), torch.zeros_like(logabsdet_0), logabsdet_0)
    logabsdet_0 = torch.where(torch.isinf(logabsdet_0), torch.zeros_like(logabsdet_0), logabsdet_0)

    logabsdet_1 = torch.where(torch.isnan(logabsdet_1), torch.zeros_like(logabsdet_1), logabsdet_1)
    logabsdet_1 = torch.where(torch.isinf(logabsdet_1), torch.zeros_like(logabsdet_1), logabsdet_1)

    # logabsdet_0 = filter_nan_inf.apply(logabsdet_0)
    # logabsdet_1 = filter_nan_inf.apply(logabsdet_1)

    C = logabsdet_1 - logabsdet_0

    N = mu_0.size(2)
    KL = 0.5 * (A + B + C - N)

    return torch.mean(KL)


# +---------------+
# | Loss function |
# +---------------+

def vae_loss_function(mu_encoder, L_encoder, mu_prior, L_prior, pos_weight, kld_multiplier = 1, norm_mu = 1, norm_sigma = 1, device = "cpu"):
    """
    Reconstruction + KL divergence losses summed over all elements and batch
    """
    batch_size = x.shape[0]

    mu_encoder = mu_encoder.permute(0, 2, 1)
    L_encoder = L_encoder.permute(0, 2, 3, 1)

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    num_nodes = 38
    num_channels = mu_encoder.size(1)

    l2_mu = norm_mu * mu_prior.norm(2)
    l2_sigma = norm_sigma / L_prior.norm(2)

    mu_prior = torch.cat([mu_prior.unsqueeze(dim = 0) for b in range(batch_size)])
    L_prior = torch.cat([L_prior.unsqueeze(dim = 0) for b in range(batch_size)])


    KLD = (kld_multiplier / num_nodes) * KL_Gaussians(mu_encoder, L_encoder, mu_prior, L_prior, device)

    return KLD + l2_mu + l2_sigma, KLD, l2_mu, l2_sigma