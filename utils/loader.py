from utils.data_loader_mol import dataloader
def load_data(config, get_graph_list=False):
    return dataloader(config, get_graph_list)