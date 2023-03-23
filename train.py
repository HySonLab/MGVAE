import torch 
import numpy as np
import time 
import torch.nn.functional as F

from utils.data_loader_mol import *
from parsers.parser import Parser 
from parsers.config import get_config
from utils.loader import load_data, 
from utils.graph_utils import mask_adjs, mask_x, node_flags
from utils.mol_utils import molecule_to_pdf
from models.vae import *
from models.vae_module import *
from models.dense import *
import torch.nn as nn
from train_utils import *

import warnings
warnings.filterwarnings("ignore")


args = Parser().parse()
config = get_config(args.config, args.seed)

torch.manual_seed(args.seed)
# Fix GPU torch random seed
torch.cuda.manual_seed(args.seed)
# Fix the Numpy random seed
np.random.seed(args.seed)

train_loader, test_loader = load_data(config)

device = "cuda:2"
encoder = GraphIsomorphismNetwork(9, [32, 32, 32], 32, "leaky_relu", epsilon = 0.5, device = device)
decoder = Decoder(encoder.fdim, 4, 9, device).to(torch.float32)
vae = SnMRF(encoder, decoder, config.data.max_node_num, device = device).to(device).to(torch.float32)
optimizer = torch.optim.AdamW(vae.parameters(), lr = 1e-3, betas=(0.9, 0.999))
total_num_batches = len(train_loader)
best_loss = 1e3
ckpt_path = config.train.ckpt_path
LOG = open("log.txt", "w")

for epoch in range(config.train.num_epochs):
    vae.train()
    print('--------------------------------------')
    print('Epoch', epoch)
    LOG.write('--------------------------------------\n')
    LOG.write('Epoch ' + str(epoch) + '\n')
    t = time.time()

    total_loss = 0.0
    total_acc = 0.0
    nBatch = 0
    
    for batch in train_loader:
        node_feat = batch[0].to(device).float()
        edge_feat = batch[1].to(device).float()
        edge_label = batch[2].to(device)
        node_label = batch[3].to(device)
        
        optimizer.zero_grad()
        flags = node_flags(node_feat)

        node_feat = mask_x(node_feat, flags)
        edge_feat = mask_adjs(edge_feat, flags)

        adj = torch.where(edge_feat.argmax(1) > 0, 1, 0).float().to(device)
        (node, edge), mu, L = vae(adj, edge_feat, node_feat, flags)
        node_label = node_label.view(-1)
        edge_label = edge_label.view(-1)

        edge = edge.permute(0, 2, 3, 1).reshape(-1, 4)
        node = node.reshape(-1, 9)
        adj = torch.sum(edge_feat, dim = 1).float().to(device)
        loss, KL, l2_mu, l2_sigma = vae_loss_function(pred_adj, adj, mu, L, vae.mu_prior, vae.L_prior, 0.5, device = device)
        
        node_loss = F.nll_loss(F.log_softmax(node, dim = -1), node_label, ignore_index = 100)
        edge_loss = F.nll_loss(F.log_softmax(edge, dim = -1), edge_label, ignore_index = 100)
       
        loss = node_loss + edge_loss + 0.1 * (KL + l2_mu + l2_sigma)
        loss.backward()
        optimizer.step()
        if nBatch % 10 == 0:
            LOG.write('Batch ' + str(nBatch) + '/' + str(total_num_batches) + ' | Loss =' + str(loss.detach().item()) + ' | Node loss = ' + str(node_loss.item()) +
                  ' | Edge loss = ' + str(edge_loss.detach().item()) +  ' | KL=' + str(KL.detach().item()) + '\n')
            print('Batch', nBatch, '/', total_num_batches, ': Loss =', loss.detach().item(), ', Node loss =', node_loss.item(), 
                  'Edge loss =', edge_loss.detach().item(), 'KL=', KL.detach().item())
        nBatch += 1
    
    
    mol_list, num_correct = vae.infer(5, device)
    for i in range(len(mol_list[:5])):
        molecule_to_pdf(mol_list[i], f"val_{i}")
        print(Chem.MolToSmiles(mol_list[i]))
    
    torch.save(vae.state_dict(), ckpt_path)
    print("--- Save checkpoint ---")
    avg_loss = total_loss / nBatch
    avg_acc = total_acc / nBatch
    print('Average loss:', avg_loss)
    LOG.write('Average loss: ' + str(avg_loss) + '\n')
    print('Average accuracy:', avg_acc)
    LOG.write('Average accuracy: ' + str(avg_acc) + '\n')
    print("Time =", "{:.5f}".format(time.time() - t))
    LOG.write("Time = " + "{:.5f}".format(time.time() - t) + "\n")

LOG.close()

