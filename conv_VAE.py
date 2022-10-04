import torch
from torch import nn
import itertools
import numpy as np
from torch.autograd import Variable
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import numpy as np

shape_log = False

class VariationalAutoEncoders(nn.Module):
    def __init__(self, INPUT_AXIS, BATCH_SIZE, LATENT_DIMENSION):
        super(VariationalAutoEncoders, self).__init__()
        self.IA = INPUT_AXIS
        self.BS = BATCH_SIZE
        self.LD = LATENT_DIMENSION
        # self.fc1_1 = nn.Conv2d(1, 100, 4, 2, 1, bias=False)
        # self.bn1_1 = nn.BatchNorm2d(100)
        # self.fc1_2 = nn.Conv2d(100, 200, 4, 2, 1, bias=False)
        # self.bn1_2 = nn.BatchNorm2d(200)
        # self.fc1_3 = nn.Conv2d(200, 300, 4, 2, 1, bias=False)
        # self.bn1_3 = nn.BatchNorm2d(300)
        # self.proj = nn.Linear(172800, 100)
        # self.fc2m   = nn.Linear(100, self.LD)
        # self.fc2s   = nn.Linear(100, self.LD)
        # self.fc3   = nn.Linear(self.LD, 100)
        # self.tr_proj = nn.Linear(100, 172800)
        # self.fc4_1 = nn.ConvTranspose2d(300, 200, 3, 2, bias=False)
        # self.bn4_1 = nn.BatchNorm2d(200)
        # self.fc4_2 = nn.ConvTranspose2d(200, 100, 2, 2, bias=False)
        # self.bn4_2 = nn.BatchNorm2d(100)
        # self.fc4_3 = nn.ConvTranspose2d(100, 1, 4, 2, bias=False)
        self.fc1_1 = nn.Conv2d(1, 100, 4, 2, 1, bias=False)
        self.bn1_1 = nn.BatchNorm2d(100)
        self.fc1_2 = nn.Conv2d(100, 200, 4, 2, 1, bias=False)
        self.bn1_2 = nn.BatchNorm2d(200)
        self.fc1_3 = nn.Conv2d(200, 300, 4, 2, 1, bias=False)
        self.bn1_3 = nn.BatchNorm2d(300)
        self.proj = nn.Linear(172800, 1000)
        self.proj2 = nn.Linear(1000, 100)
        self.fc2m   = nn.Linear(100, self.LD)
        self.fc2s   = nn.Linear(100, self.LD)
        self.fc3   = nn.Linear(self.LD, 100)
        self.tr_proj = nn.Linear(100, 1000)
        self.tr_proj2 = nn.Linear(1000, 172800)
        self.fc4_1 = nn.ConvTranspose2d(300, 200, 3, 2, bias=False)
        self.bn4_1 = nn.BatchNorm2d(200)
        self.fc4_2 = nn.ConvTranspose2d(200, 100, 2, 2, bias=False)
        self.bn4_2 = nn.BatchNorm2d(100)
        self.fc4_3 = nn.ConvTranspose2d(100, 1, 4, 2, bias=False)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        #0に平均を足してから、-x < σ < +xまでの乱数で分散を足すと正規分布の形になる
        eps = torch.randn_like(std)#平均0分散1の乱数をstdと同じテンソルの形で生成 -1~+1
        return mu + eps * std

    def encode(self, x):
        x = F.leaky_relu(self.bn1_1(self.fc1_1(x)))
        x = F.leaky_relu(self.bn1_2(self.fc1_2(x)))
        x = F.leaky_relu(self.bn1_3(self.fc1_3(x)))
        return x

    def full_connection(self, z):
        z = F.leaky_relu(self.proj(z))
        z = F.leaky_relu(self.proj2(z))
        mu, logvar = F.relu(self.fc2m(z)), F.relu(self.fc2s(z))
        return mu, logvar

    def tr_full_connection(self, z):
        y = F.leaky_relu(self.fc3(z))
        y = F.leaky_relu(self.tr_proj(y))
        y = F.leaky_relu(self.tr_proj2(y))
        return y

    def decode(self, y):
        y = F.leaky_relu(self.bn4_1(self.fc4_1(y)))
        y = F.leaky_relu(self.bn4_2(self.fc4_2(y)))
        y = torch.sigmoid(self.fc4_3(y))
        return y

    def forward(self, x):
        if shape_log == True: print(f'x:{x.shape}')
        z = self.encode(x.view(self.BS, 1, int(self.IA**0.5), int(self.IA**0.5))).cuda()
        if shape_log == True: print(f'z = encorder(x):{z.shape}')
        zs = z.shape
        mu, logvar = self.full_connection(z.view(self.BS, -1))
        reparameterized_z = self.reparameterize(mu, logvar).cuda()
        y = self.tr_full_connection(reparameterized_z).cuda()
        if shape_log == True: print(f'tr_full_connection(z):{y.shape}')
        y = self.decode(y.view(zs)).cuda()
        if shape_log == True: print(f'y:{y.shape}')
        y = y.view(self.BS, self.IA)
        return reparameterized_z, y, mu, logvar
