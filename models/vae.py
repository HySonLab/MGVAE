from models.vae_module import *
import numpy as np
import torch.nn.functional as F
from models.module import *
from utils.mol_utils import gen_mol

class SnMRF(nn.Module): 
    """
    Permutation Variational Autoencoder
    """

    def __init__(self, encoder, decoder, N, train_mu = True, train_sigma = True, device = "cpu"):
        super().__init__()

        self.encoder = encoder.to(device)
        self.decoder = decoder.to(device)

        self.train_mu = train_mu
        self.train_sigma = train_sigma
        self.device = device

        #self.fdim = self.c_hid*(self.num_layers-1) + self.c_final + self.c_init
        C = encoder.fdim

        self.params_prior = []

        if self.train_mu == True:
            self.mu_prior = torch.nn.Parameter(torch.randn(N, C, device = device, dtype = torch.float, requires_grad = True))
            self.params_prior.append(self.mu_prior)
        else:
            self.mu_prior = torch.zeros(N, C, device = device, dtype = torch.float, requires_grad = False)

        if self.train_sigma == True:
            self.L_prior = torch.nn.Parameter(torch.randn(N, N, C, device = device, dtype = torch.float, requires_grad = True))
            self.params_prior.append(self.L_prior)
        else:
            self.L_prior = torch.cat([torch.eye(N, device = device, dtype = torch.float, requires_grad = False).unsqueeze(dim = 2) for c in range(C)], dim = 2)

        self.params_prior = torch.nn.ParameterList(self.params_prior)

    def forward(self, adj, x, x_node=None, mask=None):
        # First-order & Second-order Encoder
        mu_encoder, L_encoder = self.encoder(adj, x_node, x, mask)

        mu = mu_encoder 
        L = L_encoder

        # Sigma
        sigma = torch.matmul(L, L.transpose(2, 3))

        # Reparameterization
        eps = torch.randn(mu.size()).to(device = self.device)
        # x_sample = mu + torch.einsum('bcij,bcj->bci', sigma, eps)
        x_sample = mu + torch.einsum('bcij,bcj->bci', L, eps)
        #x_sample = x_sample.transpose(1, 2)
        #return x_sample
        # Decoder
        predict = self.decoder(x_sample, mask)

        return predict, mu_encoder, L_encoder

    @torch.no_grad()
    def infer(self, num_mols, device):
        mol_lists = []
        batch_size = 32
        batch = 0
        total_correct = 0

        while batch <= num_mols:
            mu_prior = torch.cat([self.mu_prior.unsqueeze(dim = 0) for b in range(batch_size)])
            L_prior = torch.cat([self.L_prior.unsqueeze(dim = 0) for b in range(batch_size)])
            mu = mu_prior.transpose(1, 2)
            L = L_prior.transpose(2, 3).transpose(1, 2)
            eps = torch.randn(mu.size()).to(device = device)
            x_sample = mu + torch.einsum('bcij,bcj->bci', L, eps)
            node, edge, adj = self.decoder(x_sample)
            mols, num_correct =  gen_mol(node, edge, "zinc")
            mol_lists += mols 
            total_correct += num_correct
            batch += batch_size 
            print(f"Generating {batch} / {num_mols}")
        
        return mols, total_correct