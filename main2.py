#%%
import sys
import numpy as np
import random
import torch
import pandas as pd
from torch import nn
from torch.nn import functional as F
from torch import distributions as D
from torch.utils.data import DataLoader
from functions import load_data, make_directories, save_globals_and_functions, plot_latent
from coranking_matrix import get_score
import torchsort
import argparse
from differential_histgram import differentiable_histogram
from toy_AE import *
import time
from scipy.spatial import distance_matrix
from scipy.spatial.distance import pdist, squareform
import itertools
import umap
from sklearn import decomposition, manifold, neighbors
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path
from plotly import offline
from plotly.offline import init_notebook_mode, plot
#%%
#--pytorch tensor type
init_notebook_mode(connected = True)
torch.set_default_tensor_type(torch.cuda.FloatTensor)
#--set seed
seed = 10
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

#%%
parser = argparse.ArgumentParser(description='')
parser.add_argument('--data', type=str, default='qm7')
parser.add_argument('--model', type=str, default='AE')
parser.add_argument('--z_dim', type=int, default=3)
parser.add_argument('--conv', type=bool, default=False)
#-----------------------------------------------------
parser.add_argument('--epoch', type=int, default=30)
parser.add_argument('--batch_size', type=int, default=8)#8
parser.add_argument('--es', type=int, default=5)
parser.add_argument('--lr', type=float, default=1e-5)
parser.add_argument('--_lambda', type=float, default=1)
parser.add_argument('--wd', type=float, default=0)
# parser.add_argument('--epoch', type=int, default=1000)
# parser.add_argument('--batch_size', type=int, default=20)#10
# parser.add_argument('--es', type=int, default=5)
# parser.add_argument('--lr', type=float, default=1e-3)#ie-14
# parser.add_argument('--_lambda', type=float, default=1.0)
# parser.add_argument('--wd', type=float, default=0)
args = parser.parse_args()#args=[]

#%%
x_data, label = load_data(args.data, batch_size=args.batch_size, force_new=True)
label = label
# x_data = x_data[:10000]
# label = label[:10000]
#%%
def save_scores(crm_scores, elapsed_time):
    f = open(f'csv/{args.model}_{args.data}.csv', 'a')
    f.write(f'{crm_scores[0]},{crm_scores[1]},{crm_scores[2]},{crm_scores[3]},{args.z_dim},{elapsed_time}\n')
    f.close()
    return

def pairwise_norm_func(input_tensor, p):
    distances = torch.stack([torch.linalg.norm(input_tensor[n]-input_tensor[m], ord=p) for n, m in itertools.combinations(range(args.batch_size), 2)], dim=0).cuda()
    return distances

def get_mnkld(m1, v1, m2, v2):
    output = D.kl.kl_divergence(D.MultivariateNormal(m1, torch.diag_embed(v1)), D.MultivariateNormal(m2, torch.diag_embed(v2)))
    return output

def pairwise_kld_func(mu, var):
    kld = lambda m1, v1, m2, v2 : D.kl.kl_divergence(D.MultivariateNormal(m1, torch.diag_embed(v1)), D.MultivariateNormal(m2, torch.diag_embed(v2)))
    #+kld(mu[n], var[n], mu[m], var[m]))*0.5
    pairwise_kld = torch.stack([kld(mu[m], var[m], mu[n], var[n]) for n, m in itertools.combinations(range(args.batch_size), 2)], dim=0).cuda()
    return pairwise_kld

def rank_func(distance):
    return torchsort.soft_rank(distance.view(1, -1)*10000)

def get_upper_triangular(distance_matrix):
    d_m_t = distance_matrix.triu().requires_grad_().cuda()#upper triangular of d_m
    mask = torch.ones(d_m_t.shape).bool().triu(diagonal=1)
    unduplicated_distances = d_m_t[mask]#values only in upper triangular
    return unduplicated_distances

def local_distance_matrix(x_data, n_neighbors=5):
    graph = neighbors.kneighbors_graph(x_data, n_neighbors)
    graph = csr_matrix(graph)
    d_m = np.array([shortest_path(csgraph=graph, directed=False, indices=n, return_predecessors=False) for n, _ in enumerate(x_data)])
    if float('inf') in d_m:
        print('[error]n_neighbors is too small')
        sys.exit()
    return d_m

def label_distance_matrix(label, p):
    # d_m = np.zeros((len(label), len(label)))
    # for n, m in itertools.combinations(range(len(label)), 2):
    #     d_m[n, m] = np.abs(label[n]-label[m])
    d_m = squareform(pdist(label.astype(float).reshape(-1 ,1)%3))#+0.001
    return d_m

if args.conv == True:
    from conv_AE import *
    from conv_VAE import *
else:
    from linear_AE import *
    from linear_VAE import *

if args.data in ['roll', 'curve']:
    model = ToyAutoEncoders(x_data.shape[1], args.batch_size, args.z_dim).cuda()
elif args.model in ['vae', 'raw_input_and_external_dm_prob', 'raw_input_and_external_dm_prob_double']:
    model = VariationalAutoEncoders(x_data.shape[1], args.batch_size, args.z_dim).cuda()
else:
    model = AutoEncoders(x_data.shape[1], args.batch_size, args.z_dim).cuda()

mse = nn.MSELoss().cuda()#reconstruction_error
cos = nn.CosineSimilarity(dim=1, eps=1e-6)#regularization_error

if args.model =='only_raw_input':
    x_dim = x_data.shape[1]
    def custom_loss(original, latent, output):
        rec_error = mse(output, original)#reconstruction_error
        x_rank, z_rank = rank_func(pairwise_norm_func(original, 2)), rank_func(pairwise_norm_func(latent, 2))
        x2z = torch.Tensor([0])#-1 * (cos(x_rank - x_rank.mean(), z_rank - z_rank.mean()) - 1)
        reg_error = args._lambda * x2z#regularization_error
        return rec_error, reg_error
    def get_loss(data, model):
        x_batch = data.requires_grad_().cuda()
        z, y = model(x_batch)
        return custom_loss(x_batch, z, y)#comparing distance matrix(dm) of raw input and dm of latent representation
elif args.model == 'raw_input_and_external_dm':
    external_dm = distance_matrqix(x_data, x_data, p=2)#external distance matrix
    x_data = np.column_stack([x_data, np.array(list(range(len(x_data))))])
    x_dim = x_data.shape[1] - 1#without row index
    def custom_loss(original, latent, output, d_m):
        rec_error = mse(output, original)
        x_rank, z_rank = rank_func(get_upper_triangular(torch.from_numpy(d_m).type(torch.cuda.FloatTensor))), rank_func(pairwise_norm_func(latent, 2))
        x2z = -1 * (cos(x_rank - x_rank.mean(), z_rank - z_rank.mean()) - 1)
        ##############adding regularization of entire embedding
        # vib = torch.abs(0.5 - pairwise_norm_func(latent, 2).mean())
        # x2z = (x2z+vib)*0.5
        ##############
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
        x_rank, z_rank = rank_func(get_upper_triangular(torch.from_numpy(d_m).type(torch.cuda.FloatTensor))), rank_func(pairwise_norm_func(latent, 2))
        x2z = mse(x_rank, z_rank)#-1 * (cos(x_rank - x_rank.mean(), z_rank - z_rank.mean()) - 1)
        reg_error = args._lambda * x2z
        return rec_error, reg_error
    def get_loss(data, model):
        batch_idx = list(map(int, data[:, x_dim].tolist()))
        d_m = data[:, batch_idx]
        data = data[:, :x_dim].detach()
        x_batch = data.requires_grad_().cuda()
        z, y = model(x_batch)
        return custom_loss(x_batch, z, y, d_m)#comparing dm of latent representation and dm as raw input
    def get_loss(data, model):
        batch_idx = list(map(int, data[:, x_dim].tolist()))#index on each batch
        d_m = external_dm[batch_idx][:, batch_idx]#indexed d_m
        data = data[:, :x_dim]#without row index column
        x_batch = data.requires_grad_().cuda()
        z, y = model(x_batch)
        return custom_loss(x_batch, z, y, d_m)#comparing dm of latent representation and external_dm
elif args.model == 'vae':
    x_dim = x_data.shape[1]
    def custom_loss(original, latent, output, mu, logvar):
        rec_error = F.binary_cross_entropy_with_logits(output, original.view(-1, x_data.shape[1]), reduction='sum')
        reg_error = 0.5 * torch.sum(mu**2 + logvar.exp() - logvar - 1)
        return rec_error, reg_error
    def get_loss(data, model):
        x_batch = data.requires_grad_().cuda()
        z, y, mu, logvar = model(x_batch)
        return custom_loss(x_batch, z, y, mu, logvar)#comparing distance matrix(dm) of raw input and dm of latent representation
elif args.model == 'raw_input_and_external_dm_prob':
    external_dm = label_distance_matrix(label, p=2)#local_distance_matrix(x_data, 100)#distance_matrix(x_data, x_data, p=2)#
    x_data = np.column_stack([x_data, np.array(list(range(len(x_data))))])
    x_dim = x_data.shape[1] - 1
    def custom_loss(original, latent, output, mu, logvar, d_m):
        rec_error = mse(output, original)
        pkld = pairwise_kld_func(mu, logvar.exp())
        x_rank, z_rank = rank_func(get_upper_triangular(torch.from_numpy(d_m).type(torch.cuda.FloatTensor))), rank_func(pkld)
        x2z = -1 * (cos(x_rank - x_rank.mean(), z_rank - z_rank.mean()) - 1)
        vib = torch.abs(0.5 - pairwise_norm_func(mu, 2).mean())#0.5 * torch.sum(mu**2 + logvar.exp() - logvar - 1)#
        reg_error = args._lambda * ((x2z+vib)*0.5)
        return rec_error, reg_error
    def get_loss(data, model):
        batch_idx = list(map(int, data[:, x_dim].tolist()))#index on each batch
        d_m = label_distance_matrix(label[batch_idx], p=2)#external_dm[batch_idx][:, batch_idx]#indexed d_m
        data = data[:, :x_dim]#without row index column
        x_batch = data.requires_grad_().cuda()
        z, y, mu, logvar = model(x_batch)
        return custom_loss(x_batch, z, y, mu, logvar, d_m)#comparing distance matrix(dm) of raw input and dm of latent representation
elif args.model == 'raw_input_and_external_dm_prob_double':
    x_data = np.column_stack([x_data, np.array(list(range(len(x_data))))])
    x_dim = x_data.shape[1] - 1
    def custom_loss(original, latent, output, mu, logvar, d_m1, d_m2):
        rec_error = mse(output, original)
        pkld = pairwise_kld_func(mu, logvar.exp())
        x_rank1, z_rank1 = rank_func(get_upper_triangular(torch.from_numpy(d_m1).type(torch.cuda.FloatTensor))), rank_func(pkld)
        x2z1 = -1 * (cos(x_rank1 - x_rank1.mean(), z_rank1 - z_rank1.mean()) - 1)
        x_rank2, z_rank2 = rank_func(get_upper_triangular(torch.from_numpy(d_m2).type(torch.cuda.FloatTensor))), rank_func(torch.linalg.norm(mu, ord=2))
        x2z2 = -1 * (cos(x_rank2 - x_rank2.mean(), z_rank2 - z_rank2.mean()) - 1)
        vib = torch.abs(0.5 - pairwise_norm_func(mu, 2).mean())#0.5 * torch.sum(mu**2 + logvar.exp() - logvar - 1)#
        reg_error = args._lambda * ((x2z1+x2z2+vib)*0.5)#
        return rec_error, reg_error
    def get_loss(data, model):
        batch_idx = list(map(int, data[:, x_dim].tolist()))#index on each batch
        d_m1 = distance_matrix(x_data[batch_idx], x_data[batch_idx], p=2)
        d_m2 = distance_matrix(label[batch_idx].reshape(-1, 1), label[batch_idx].reshape(-1, 1), p=2)#distance_matrix(label[batch_idx].reshape(-1, 1), np.zeros(label[batch_idx].shape).reshape(-1, 1), p=2)
        data = data[:, :x_dim]#without row index column
        x_batch = data.requires_grad_().cuda()
        z, y, mu, logvar = model(x_batch)
        return custom_loss(x_batch, z, y, mu, logvar, d_m1, d_m2)#comparing distance matrix(dm) of raw input and dm of latent representation
elif args.model == 'umap':
    x_dim = x_data.shape[1]
    start_time = time.time()
    lat_result = np.array(umap.UMAP(n_components=args.z_dim).fit_transform(x_data))
    crm_scores = np.round(get_score(x_data[:, :x_dim], lat_result), decimals=5)
    print(f'Lt:{crm_scores[0]}\nLs:{crm_scores[1]}\nGt:{crm_scores[2]}\nGs:{crm_scores[3]}\n ')
    elapsed_time = time.time() - start_time
    save_scores(crm_scores, elapsed_time)
    fig = plot_latent(lat_result, label, np.repeat(5, len(lat_result)), 'scatter', f'scatter_{args.data}_{args.model}')
    fig.write_image(f"comparison//scatter_{args.data}_{args.model}.png")
    plot(fig, show_link = True, filename = f'html//scatter_{args.data}_{args.model}.html')
    sys.exit()
elif args.model == 'pca':
    x_dim = x_data.shape[1]
    start_time = time.time()
    lat_result = np.array(decomposition.PCA(n_components=args.z_dim).fit_transform(x_data))
    crm_scores = np.round(get_score(x_data[:, :x_dim], lat_result), decimals=5)
    print(f'Lt:{crm_scores[0]}\nLs:{crm_scores[1]}\nGt:{crm_scores[2]}\nGs:{crm_scores[3]}\n ')
    elapsed_time = time.time() - start_time
    save_scores(crm_scores, elapsed_time)
    fig = plot_latent(lat_result, label, np.repeat(5, len(lat_result)), 'scatter', f'scatter_{args.data}_{args.model}')
    fig.write_image(f"comparison//scatter_{args.data}_{args.model}.png")
    plot(fig, show_link = True, filename = f'html//scatter_{args.data}_{args.model}.html')
    sys.exit()
elif args.model == 'mds':
    x_dim = x_data.shape[1]
    start_time = time.time()
    lat_result = np.array(manifold.MDS(n_components=args.z_dim).fit_transform(x_data))
    crm_scores = np.round(get_score(x_data[:, :x_dim], lat_result), decimals=5)
    print(f'Lt:{crm_scores[0]}\nLs:{crm_scores[1]}\nGt:{crm_scores[2]}\nGs:{crm_scores[3]}\n ')
    elapsed_time = time.time() - start_time
    save_scores(crm_scores, elapsed_time)
    sys.exit()

#%%
optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
print(model)
make_directories()
save_globals_and_functions(vars(args), [pairwise_norm_func, rank_func, custom_loss, get_loss])

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
        #torch.autograd.set_detect_anomaly(True)
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
model.eval()
if args.model in ['vae', 'raw_input_and_external_dm_prob']:
    lat_result = np.empty((0, args.z_dim))
    lat_var = np.empty((0, args.z_dim))
    for n, data in enumerate(DataLoader(torch.from_numpy(x_data).type(torch.cuda.FloatTensor), batch_size = args.batch_size, shuffle = False)):#シャッフルしない
        batch = data[:, :x_dim].view(args.batch_size, 1, x_dim).cuda()
        temp = model(batch)
        lat_result = np.vstack([lat_result, temp[2].view(args.batch_size, args.z_dim).data.cpu().numpy()])#numpy().reshape(args.batch_size, args.z_dim)
        lat_var = np.vstack([lat_var, torch.exp(temp[3]).view(args.batch_size, args.z_dim).data.cpu().numpy()])
    print(lat_result.mean(),lat_var.mean())
    print(lat_result.var(),lat_var.var())
    fig = plot_latent(lat_result, label, lat_var[:, 1], 'scatter', f'scatter_{args.data}_{args.model}')
    fig.write_image(f"comparison//scatter_{args.data}_{args.model}.png")
    plot(fig, show_link = True, filename = f'html//scatter_{args.data}_{args.model}.html')
    fig2 = plot_latent(lat_result, label, lat_var[:, 1], 'density', f'density_{args.data}_{args.model}')
    fig2.write_image(f"comparison//density_{args.data}_{args.model}.png")
    plot(fig2, show_link = True, filename = f'html//density_{args.data}_{args.model}.html')
    fig3 = plot_latent(lat_result, label, lat_var[:, 1], 'stacking2d', f'stacking2d_{args.data}_{args.model}')
    fig3.write_image(f"comparison//stacking2d_{args.data}_{args.model}.png")
    plot(fig3, show_link = True, filename = f'html//stacking2d_{args.data}_{args.model}.html')
else:
    lat_result = np.empty((0, args.z_dim))
    for n, data in enumerate(DataLoader(torch.from_numpy(x_data).type(torch.cuda.FloatTensor), batch_size = args.batch_size, shuffle = False)):#シャッフルしない
        batch = data[:, :x_dim].view(args.batch_size, 1, x_dim).cuda()
        temp = model(batch)
        lat_result = np.vstack([lat_result, temp[0].view(args.batch_size, args.z_dim).data.cpu().numpy()])#numpy().reshape(args.batch_size, args.z_dim)
    fig = plot_latent(lat_result, label, np.repeat(5, len(lat_result)), 'scatter', f'scatter_{args.data}_{args.model}')
    fig.write_image(f"comparison//scatter_{args.data}_{args.model}.png")
    plot(fig, show_link = True, filename = f'html//scatter_{args.data}_{args.model}.html')
    fig2 = plot_latent(lat_result, label, np.repeat(5, len(lat_result)), 'density', f'density_{args.data}_{args.model}')
    fig2.write_image(f"comparison//density_{args.data}_{args.model}.png")
    plot(fig2, show_link = True, filename = f'html//density_{args.data}_{args.model}.html')
    fig3 = plot_latent(lat_result, label, np.repeat(5, len(lat_result)), 'stacking2d', f'stacking2d_{args.data}_{args.model}')
    fig3.write_image(f"comparison//stacking2d_{args.data}_{args.model}.png")
    plot(fig3, show_link = True, filename = f'html//stacking2d_{args.data}_{args.model}.html')

#%%
elapsed_time = time.time() - start_time
crm_scores = [0,0,0,0]#np.round(get_score(x_data[:, :x_dim], lat_result), decimals=5)
save_scores(crm_scores, elapsed_time)
print(f'Lt:{crm_scores[0]}\nLs:{crm_scores[1]}\nGt:{crm_scores[2]}\nGs:{crm_scores[3]}\n ')

#%%
np.savetxt('out.csv', lat_result, delimiter=',')