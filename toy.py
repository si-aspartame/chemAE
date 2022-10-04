#%%
import sys
import numpy as np
import random
import torch
import pandas as pd
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from functions import load_data, make_directories, save_globals_and_functions, plot_latent
from coranking_matrix import get_score
import torchsort
import argparse
from toy_AE import *
from linear_VAE import *
import time
from scipy.spatial import distance_matrix
import itertools
import umap
from sklearn import decomposition, manifold, neighbors
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path

#%%
torch.set_default_tensor_type(torch.cuda.FloatTensor)
#--set seed
seed = 10
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

#%%
parser = argparse.ArgumentParser(description='')
parser.add_argument('--data', type=str, default='roll')
parser.add_argument('--model', type=str, default='only_raw_input')
parser.add_argument('--z_dim', type=int, default=2)
#-----------------------------------------------------
parser.add_argument('--epoch', type=int, default=100)
parser.add_argument('--batch_size', type=int, default=10)#10
parser.add_argument('--es', type=int, default=5)
parser.add_argument('--lr', type=float, default=1e-2)
parser.add_argument('--_lambda', type=float, default=10)#100
parser.add_argument('--wd', type=float, default=0)
args = parser.parse_args(args=[])#

#%%
x_data, label = load_data('roll', batch_size=1)
# x_data[:, 1] = x_data[:, 1]*0.5
plot_latent(x_data, label).show()
#%%

def local_distance_matrix(x_data):
    graph = neighbors.kneighbors_graph(x_data, 5)
    graph = csr_matrix(graph)
    d_m = np.array([shortest_path(csgraph=graph, directed=False, indices=n, return_predecessors=False) for n,_ in enumerate(x_data)])
    return d_m

#%%
def save_scores(crm_scores, elapsed_time):
    f = open(f'csv/{args.model}_{args.data}.csv', 'a')
    f.write(f'{crm_scores[0]},{crm_scores[1]},{crm_scores[2]},{crm_scores[3]},{args.z_dim},{elapsed_time}\n')
    f.close()
    return

def distance_func(input_tensor):
    return torch.stack([torch.linalg.norm(input_tensor[n[0]]-input_tensor[n[1]]) for n in itertools.combinations(range(args.batch_size), 2)], dim=0).cuda()

def rank_func(distance):
    return torchsort.soft_rank(distance.view(1, -1))

def get_upper_triangular(distance_matrix):
    d_m_t = torch.from_numpy(distance_matrix).type(torch.cuda.FloatTensor).triu().requires_grad_().cuda()#upper triangular of d_m
    mask = torch.ones(d_m_t.shape).bool().triu(diagonal=1)
    unduplicated_distances = d_m_t[mask]#values only in upper triangular
    return unduplicated_distances

model = AutoEncoders(x_data.shape[1], args.batch_size, args.z_dim).cuda()
mse = nn.MSELoss().cuda()#reconstruction_error
cos = nn.CosineSimilarity(dim=1, eps=1e-6)#regularization_error

if args.model =='only_raw_input':
    x_dim = x_data.shape[1]
    def custom_loss(original, latent, output):
        rec_error = mse(output, original)#reconstruction_error
        x_rank, z_rank = rank_func(distance_func(original)), rank_func(distance_func(latent))
        x2z = -1 * (cos(x_rank - x_rank.mean(), z_rank - z_rank.mean()) - 1)
        reg_error = args._lambda * x2z#regularization_error
        return rec_error, reg_error
    def get_loss(data, model):
        x_batch = data.requires_grad_().cuda()
        z, y = model(x_batch)
        return custom_loss(x_batch, z, y)#comparing distance matrix(dm) of raw input and dm of latent representation
elif args.model == 'raw_input_and_external_dm':
    external_dm = distance_matrix(x_data, x_data, p=2)#local_distance_matrix(x_data, label)#
    x_data = np.column_stack([x_data, np.array(list(range(len(x_data))))])
    x_dim = x_data.shape[1] - 1#without row index
    def custom_loss(original, latent, output, d_m):
        rec_error = mse(output, original)
        x_rank, z_rank = rank_func(get_upper_triangular(d_m)), rank_func(distance_func(latent))
        x2z = -1 * (cos(x_rank - x_rank.mean(), z_rank - z_rank.mean()) - 1)
        reg_error = args._lambda * x2z
        return rec_error, reg_error
    def get_loss(data, model):
        batch_idx = list(map(int, data[:, x_dim].tolist()))#index on each batch
        d_m = external_dm[batch_idx][:, batch_idx]#indexed d_m
        data = data[:, :x_dim]#without row index column
        x_batch = data.requires_grad_().cuda()
        z, y = model(x_batch)
        return custom_loss(x_batch, z, y, d_m)#comparing dm of latent representation and external_dm
elif args.model == 'only_dm':
    x_data = np.column_stack([x_data, np.array(list(range(len(x_data))))])#when using only dm, x_data is dm
    x_dim = x_data.shape[1] - 1#without row index
    def custom_loss(original, latent, output, d_m):
        rec_error = mse(output, original)
        x_rank, z_rank = rank_func(get_upper_triangular(d_m)), rank_func(distance_func(latent))
        x2z = -1 * (cos(x_rank - x_rank.mean(), z_rank - z_rank.mean()) - 1)
        reg_error = args._lambda * x2z
        return rec_error, reg_error
    def get_loss(data, model):
        batch_idx = list(map(int, data[:, x_dim].tolist()))
        d_m = data[:, batch_idx]
        data = data[:, :x_dim].detach()
        x_batch = data.requires_grad_().cuda()
        z, y = model(x_batch)
        return custom_loss(x_batch, z, y, d_m)#comparing dm of latent representation and dm as raw input
elif args.model == 'vae':
    x_dim = x_data.shape[1]
    model = VariationalAutoEncoders(x_data.shape[1], args.batch_size, args.z_dim).cuda()
    def custom_loss(original, latent, output, mu, logvar):
        rec_error = F.binary_cross_entropy_with_logits(output, original.view(-1, x_data.shape[1]), reduction='sum')
        reg_error = 0.5 * torch.sum(mu**2 + logvar.exp() - logvar - 1)
        return rec_error, reg_error
    def get_loss(data, model):
        x_batch = data.requires_grad_().cuda()
        z, y, mu, logvar = model(x_batch)
        return custom_loss(x_batch, z, y, mu, logvar)#comparing distance matrix(dm) of raw input and dm of latent representation
elif args.model == 'umap':
    x_dim = x_data.shape[1]
    start_time = time.time()
    lat_result = np.array(umap.UMAP(n_components=args.z_dim).fit_transform(x_data))
    crm_scores = np.round(get_score(x_data[:, :x_dim], lat_result), decimals=5)
    print(f'Lt:{crm_scores[0]}\nLs:{crm_scores[1]}\nGt:{crm_scores[2]}\nGs:{crm_scores[3]}\n ')
    elapsed_time = time.time() - start_time
    save_scores(crm_scores, elapsed_time)
    if args.z_dim < 4:
        plot_latent(lat_result, label).write_image(f'comparison/{args.z_dim}_{args.model}.png')
    sys.exit()
elif args.model == 'pca':
    x_dim = x_data.shape[1]
    start_time = time.time()
    lat_result = np.array(decomposition.PCA(n_components=args.z_dim).fit_transform(x_data))
    crm_scores = np.round(get_score(x_data[:, :x_dim], lat_result), decimals=5)
    print(f'Lt:{crm_scores[0]}\nLs:{crm_scores[1]}\nGt:{crm_scores[2]}\nGs:{crm_scores[3]}\n ')
    elapsed_time = time.time() - start_time
    save_scores(crm_scores, elapsed_time)
    if args.z_dim < 4:
        plot_latent(lat_result, label).write_image(f'comparison/{args.z_dim}_{args.model}.png')
    sys.exit()
elif args.model == 'mds':
    x_dim = x_data.shape[1]
    start_time = time.time()
    lat_result = np.array(manifold.MDS(n_components=args.z_dim).fit_transform(x_data))
    crm_scores = np.round(get_score(x_data[:, :x_dim], lat_result), decimals=5)
    print(f'Lt:{crm_scores[0]}\nLs:{crm_scores[1]}\nGt:{crm_scores[2]}\nGs:{crm_scores[3]}\n ')
    elapsed_time = time.time() - start_time
    save_scores(crm_scores, elapsed_time)
    if args.z_dim < 4:
        plot_latent(lat_result, label).write_image(f'comparison/{args.z_dim}_{args.model}.png')
    sys.exit()

#%%
optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
print(model)
make_directories()
save_globals_and_functions(vars(args), [distance_func, rank_func, custom_loss, get_loss])

#%%
all_loss = np.array([])
best_loss = 99999
es_count = 0
start_time = time.time()
print(f"final_length:{len(x_data)}")
#%%
for epoch in range(1, args.epoch+1):
    temp_loss = np.array([])#loss, rec_error, reg_error, x2y, x2z
    model.train()
    for n, data in enumerate(DataLoader(torch.from_numpy(x_data).type(torch.cuda.FloatTensor), batch_size = args.batch_size, shuffle = True, generator=torch.Generator(device='cuda'))):
        rec_error, reg_error = get_loss(data, model)
        loss = rec_error + reg_error
        optimizer.zero_grad()
        loss.backward()#######################################################################retain_graph=True
        optimizer.step()
        temp_loss = np.append(temp_loss, np.array(list(map(lambda tensor:tensor.data.sum().item() / (len(x_data) / args.batch_size), [loss, rec_error, reg_error])))).reshape(-1, 3)
    temp_loss = np.sum(temp_loss, axis=0)
    loss_dict = {'loss':temp_loss[0], 'rec_error':temp_loss[1], 'reg_error':temp_loss[2]}
    if loss_dict['loss'] < best_loss:
        print(f'[BEST] ', end='')
        torch.save(model.state_dict(), f'best.pth')
        best_loss = loss_dict['loss']
        es_count = 0
    es_count += 1
    print(f"epoch [{epoch}/{args.epoch}], loss:{loss_dict['loss']}, {int(time.time()-start_time)}s \n rec_error = {loss_dict['rec_error']}, reg_error:{loss_dict['reg_error']}")
    all_loss = np.append(all_loss, np.array([epoch, loss_dict['loss'], loss_dict['rec_error'], loss_dict['reg_error']])).reshape(-1, 4)
    if es_count == args.es:
        print('early stopping!')
        break#early_stopping

#%%
model.load_state_dict(torch.load(f'best.pth'))
lat_result = np.empty((0, args.z_dim))
model.eval()
for n, data in enumerate(DataLoader(torch.from_numpy(x_data).type(torch.cuda.FloatTensor), batch_size = args.batch_size, shuffle = False)):#シャッフルしない
    batch = data[:, :x_dim].view(args.batch_size, 1, x_dim).cuda()
    temp = model(batch)
    lat_result = np.vstack([lat_result, temp[0].view(args.batch_size, args.z_dim).data.cpu().numpy()])#numpy().reshape(args.batch_size, args.z_dim)

#%%
elapsed_time = time.time() - start_time
crm_scores = np.round(get_score(x_data[:, :x_dim], lat_result), decimals=5)
save_scores(crm_scores, elapsed_time)
if args.z_dim < 4:
    plot_latent(lat_result, label).write_image(f'comparison/{args.z_dim}_{args.model}.png')
print(f'Lt:{crm_scores[0]}\nLs:{crm_scores[1]}\nGt:{crm_scores[2]}\nGs:{crm_scores[3]}\n ')

#%%
# plot_latent(lat_result, label)