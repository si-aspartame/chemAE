import torch
from torch import nn
import torch.nn.functional as F

shape_log = False

class VariationalAutoEncoders(nn.Module):
    def __init__(self, INPUT_AXIS, BATCH_SIZE, LATENT_DIMENSION):
        super(VariationalAutoEncoders, self).__init__()
        self.IA = INPUT_AXIS
        self.BS = BATCH_SIZE
        self.LD = LATENT_DIMENSION
        #FMNIST
        # self.fc1_1 = nn.Linear(self.IA, 400)
        # self.fc1_2 = nn.Linear(400, 200)
        # self.fc1_3 = nn.Linear(200, 50)
        # self.fc2   = nn.Linear(50, self.LD)
        # self.fc3   = nn.Linear(self.LD, 50)
        # self.fc4_1 = nn.Linear(50, 200)
        # self.fc4_2 = nn.Linear(200, 400)
        # self.fc4_3 = nn.Linear(400, self.IA)
        #QM7
        self.unit = int((self.IA-self.LD)/4)+40
        self.fc1_1 = nn.Linear(self.IA, self.unit*3)
        self.fc1_2 = nn.Linear(self.unit*3, self.unit)
        # self.fc1_3 = nn.Linear(self.unit*2, self.unit)
        self.fc2m  = nn.Linear(self.unit, self.LD)
        self.fc2s  = nn.Linear(self.unit, self.LD)
        self.fc3   = nn.Linear(self.LD, self.unit)
        # self.fc4_1 = nn.Linear(self.unit, self.unit*2)
        self.fc4_2 = nn.Linear(self.unit, self.unit*3)
        self.fc4_3 = nn.Linear(self.unit*3, self.IA)

    def encoder(self, x):
        x = self.fc1_1(x)
        x = torch.relu(x)
        x = self.fc1_2(x)
        x = torch.relu(x)
        # x = self.fc1_3(x)
        # x = torch.relu(x)
        return self.fc2m(x), self.fc2s(x)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        #0に平均を足してから、-x < σ < +xまでの乱数で分散を足すと正規分布の形になる
        eps = torch.randn_like(std)#平均0分散1の乱数をstdと同じテンソルの形で生成 -1~+1
        return mu + eps * std

    def full_connection(self, z):
        z = self.fc2(z)
        return z

    def tr_full_connection(self, y):
        y = torch.relu(y)
        y = self.fc3(y)
        y = torch.relu(y)
        return y

    def decoder(self, y):
        # y = self.fc4_1(y)
        # y = torch.relu(y)
        y = self.fc4_2(y)
        y = torch.relu(y)
        y = self.fc4_3(y)
        y = torch.sigmoid(y)###tanh
        return y
    
    def forward(self, x):
        if shape_log == True: print(f'x:{x.shape}')
        x = x.view(-1, self.IA)
        mu, logvar = self.encoder(x)
        # print(f'mu:{mu}')
        # print(f'logvar:{logvar}')
        z = self.reparameterize(mu, logvar)
        if shape_log == True: print(f'z = encorder(x):{z.shape}')
        #----------------------------------------
        y = self.tr_full_connection(z).cuda()
        if shape_log == True: print(f'tr_full_connection(z):{y.shape}')
        y = self.decoder(y).view(self.BS, self.IA).cuda()
        if shape_log == True: print(f'y:{y.shape}')
        y = y.view(self.BS, self.IA)##
        #lat_repr = z.view(self.BS, self.LD).cuda()
        return z, y, mu, logvar
