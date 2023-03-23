from utils.data_loader_mol import *
from parsers.parser import Parser 
from parsers.config import get_config
from utils.loader import load_data, load_model_params, load_model
from utils.graph_utils import mask_adjs, mask_x, node_flags
from models.ScoreNetwork_X import ScoreNetworkX_GMH
from models.ScoreNetwork_A import ScoreNetworkA
from models.vae import *
from models.vae_module import *
import torch.nn as nn
from train_utils import *
from models.dense import *
import torch 
import numpy as np
import time 
from rdkit import Chem
from rdkit.Chem import Draw
import cairosvg
from rdkit.Chem.Draw import rdMolDraw2D
import random
from utils.mol_utils import molecule_to_pdf
args = Parser().parse()
config = get_config(args.config, args.seed)


#encoder = Encoder(param_adj, device).to(torch.float32)
device = "cuda:0"
encoder = GraphIsomorphismNetwork(9, [32, 32, 32], 32, "leaky_relu", epsilon = 0.5, device = device)
decoder = Decoder(encoder.fdim, 4, 9, device).to(torch.float32)
vae = SnMRF(encoder, decoder, config.data.max_node_num, device = device).to(device).to(torch.float32)
ckpt_path = config.train.ckpt_path
checkpoint = torch.load(ckpt_path)
vae.load_state_dict(checkpoint)
mol_list, num_correct = vae.infer(100, device)
print(f"Validity {100 - num_correct} / 100")
for i in range(len(mol_list[:10])):
    molecule_to_pdf(mol_list[i], f"high_order_{i}")
    print(Chem.MolToSmiles(mol_list[i]))