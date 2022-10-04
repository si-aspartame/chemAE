import torch
from torch import nn
import itertools
import numpy as np
from torch.autograd import Variable
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import numpy as np

shape_log = False

class AutoEncoders(nn.Module):
    def __init__(self, INPUT_AXIS, BATCH_SIZE, LATENT_DIMENSION):
        super(AutoEncoders, self).__init__()
        self.IA = INPUT_AXIS
        self.BS = BATCH_SIZE
        self.LD = LATENT_DIMENSION
        self.fc1_1 = nn.Conv2d(1, 100, 4, 2, 1, bias=False)
        self.bn1_1 = nn.BatchNorm2d(100)
        self.fc1_2 = nn.Conv2d(100, 200, 4, 2, 1, bias=False)
        self.bn1_2 = nn.BatchNorm2d(200)
        self.fc1_3 = nn.Conv2d(200, 300, 4, 2, 1, bias=False)
        self.bn1_3 = nn.BatchNorm2d(300)
        self.proj = nn.Linear(2700, 100)
        self.fc2   = nn.Linear(100, self.LD)
        self.fc3   = nn.Linear(self.LD, 100)
        self.tr_proj = nn.Linear(100, 2700)
        self.fc4_1 = nn.ConvTranspose2d(300, 200, 3, 2, bias=False)
        self.bn4_1 = nn.BatchNorm2d(200)
        self.fc4_2 = nn.ConvTranspose2d(200, 100, 2, 2, bias=False)
        self.bn4_2 = nn.BatchNorm2d(100)
        self.fc4_3 = nn.ConvTranspose2d(100, 1, 2, 2, bias=False)

    def make_distance_vector(self, input_tensor):
        #input_diff_sum = torch.stack([torch.linalg.norm(input_tensor[n[0]]-input_tensor[n[1]]) for n in itertools.combinations(range(self.BS), 2)], dim=0).cuda()
        triu_matrix = torch.cdist(input_tensor, input_tensor).triu().view(-1).cuda()
        input_diff_sum = triu_matrix[triu_matrix.nonzero().view(-1)].cuda()
        return input_diff_sum

    def encoder(self, x):
        x = F.leaky_relu(self.bn1_1(self.fc1_1(x)))
        x = F.leaky_relu(self.bn1_2(self.fc1_2(x)))
        x = F.leaky_relu(self.bn1_3(self.fc1_3(x)))
        return x

    def full_connection(self, z):
        z = F.leaky_relu(self.proj(z))###
        z = torch.sigmoid(self.fc2(z))
        return z

    def tr_full_connection(self, z):
        y = F.leaky_relu(self.fc3(z))
        y = F.leaky_relu(self.tr_proj(y))
        return y

    def decoder(self, y):
        y = F.leaky_relu(self.bn4_1(self.fc4_1(y)))
        y = F.leaky_relu(self.bn4_2(self.fc4_2(y)))
        y = torch.sigmoid(self.fc4_3(y))
        return y
    
    def forward(self, x):
        if shape_log == True: print(f'x:{x.shape}')
        x = x.view(self.BS, 1, int(self.IA**0.5), int(self.IA**0.5))#x = x.view(-1, self.IA)
        z = self.encoder(x).cuda()
        zs = z.shape##
        if shape_log == True: print(f'z = encorder(x):{z.shape}')
        z = z.view(self.BS, -1)
        z = self.full_connection(z).cuda()#self.full_connection(torch.tanh(z)).cuda()
        if shape_log == True: print(f'full_connection(z):{z.shape}')
        #----------------------------------------
        y = self.tr_full_connection(z).cuda()#self.tr_full_connection(z).cuda()
        if shape_log == True: print(f'tr_full_connection(z):{y.shape}')
        y = self.decoder(y.view(zs)).cuda()#self.decoder(y).view(self.BS, self.IA).cuda()
        if shape_log == True: print(f'y:{y.shape}')
        y = y.view(self.BS, self.IA)##
        #-----------------------------------------
        in_diff_sum = self.make_distance_vector(x.view(self.BS, self.IA))
        lat_diff_sum = self.make_distance_vector(z.view(self.BS, self.LD))
        out_diff_sum = self.make_distance_vector(y.view(self.BS, self.IA))
        lat_repr = z.view(self.BS, self.LD).cuda()
        return y, in_diff_sum, lat_diff_sum, out_diff_sum, lat_repr