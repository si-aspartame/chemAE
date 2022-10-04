import numpy as np
from multiprocessing import Pool
import multiprocessing as multi
from sklearn import metrics
from scipy.stats import rankdata
import gc
import pandas as pd
import time
import random
#これはオリジナル版、厳密なCRMを書き直して公開したい
L_default = {'n_iter':4000,'ks':10,'kt':1, 'i_idx':0,'j_idxes':[],}#近傍10個#4
G_default = {'n_iter':4000,'ks':1000,'kt':10,'i_idx':0,'j_idxes':[],}#ktは誤差、ランダムに1000個#40

def get_rank(i, j):
    return rankdata(np.linalg.norm(j-i, axis=1), method='dense')

def get_neighbors(input_vector, i_idx, n_neighbors):
    idx_dis = np.array([np.arange(len(input_vector)), np.linalg.norm(input_vector-input_vector[i_idx], axis=1)]).T
    sorted_idx_dis = idx_dis[np.argsort(idx_dis[:, 1])][1:n_neighbors+1]
    return sorted_idx_dis[:, 0].astype(int)

def evaluate(input_data, reduced_data, i_idx, kt, j_idxes):
    R_i, R_j = input_data[i_idx], input_data[j_idxes]
    r_i, r_j = reduced_data[i_idx], reduced_data[j_idxes]
    R_rank, r_rank = get_rank(R_i, R_j), get_rank(r_i, r_j)
    count = np.count_nonzero(np.abs(R_rank - r_rank) < kt)
    return count

def get_score(input_data, reduced_data, Local=L_default, Global=G_default, strict=False):
    Local_score = 0
    Global_score = 0
    index_list = np.arange(len(input_data))
    Local_start_time = time.time()
    if strict == True:
        L_ref_points = np.arange(len(input_data))
        G_ref_points = np.arange(len(input_data))
    else:
        L_ref_points = random.sample(list(range(len(input_data))), Local['n_iter'])
        G_ref_points = random.sample(list(range(len(input_data))), Global['n_iter'])
    for n, d in enumerate(L_ref_points):
        #print(f'[{n}]', end='')
        Local['i_idx'] = d
        Local['j_idxes'] = get_neighbors(input_data, Local['i_idx'], Local['ks'])
        temp_Local_score = evaluate(input_data, reduced_data, Local['i_idx'], Local['kt'], Local['j_idxes'])
        Local_score += temp_Local_score / (Local['ks']*len(L_ref_points))
    Local_time = time.time() - Local_start_time
    Global_start_time = time.time()
    for n, d in enumerate(G_ref_points):
        #print(f'[{n}]', end='')
        Global['i_idx'] = d
        Global['j_idxes'] = np.random.choice(np.arange(len(input_data)), size=Global['ks'])
        temp_Global_score = evaluate(input_data, reduced_data, Global['i_idx'], Global['kt'], Global['j_idxes'])
        Global_score += temp_Global_score / (Global['ks']*len(G_ref_points))
    Global_time = time.time() - Global_start_time
    return [Local_time, Local_score, Global_time, Global_score]