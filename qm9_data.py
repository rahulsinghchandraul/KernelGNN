import os
import numpy as np
import scipy.sparse as sp
import torch
import dgl

from dgl.dataloading import GraphDataLoader

from tqdm import trange
from dgl.data import QM9Dataset
from dgl.data.utils import load_graphs, save_graphs
from dgl.convert import graph as dgl_graph

from torch.utils.data.sampler import SubsetRandomSampler


data = QM9Dataset(label_keys=['gap'],  cutoff=5.0) 

N = len(data) 

N1 = 20000 # Number of graphs to use from the full dataset (130k)



full_sampler =  SubsetRandomSampler(torch.arange(N1))

full_dataloader = GraphDataLoader(data, sampler=full_sampler, batch_size = N1, drop_last = False)

it = iter(full_dataloader)
full_batch = next(it)

g_batch, labels = full_batch

src, dst = g_batch.edges()
src = np.asarray(src) + 1
dst = np.asarray(dst) + 1
adj_lst = np.column_stack((src, dst))

# print(adj_lst)
print(g_batch)
# print(src)

np.savetxt('QM9_A.txt', adj_lst, fmt='%d', delimiter=',')

np.savetxt('QM9_graph_labels.txt', labels, fmt='%1.4e')

# print(labels)


print('#nodes for each graph in the batch:', g_batch.batch_num_nodes())

rep_nodes = np.asarray(g_batch.batch_num_nodes())

graph_id = np.arange(1, N1+1, 1, dtype=int)

# print(graph_id)

graph_indicator = np.repeat(graph_id, rep_nodes)

print(graph_indicator)

np.savetxt('QM9_graph_indicator.txt', graph_indicator, fmt='%d')

