import torch
from torch import nn
import torch.nn.functional as F

shape_log = False

class ToyAutoEncoders(nn.Module):
    def __init__(self, INPUT_AXIS, BATCH_SIZE, LATENT_DIMENSION):
        super(ToyAutoEncoders, self).__init__()
        self.IA = INPUT_AXIS
        self.BS = BATCH_SIZE
        self.LD = LATENT_DIMENSION

        self.unit = int((self.IA-self.LD)/4)+10
        self.fc1_1 = nn.Linear(self.IA, self.unit*3)
        self.fc1_2 = nn.Linear(self.unit*3, self.unit*2)
        self.fc1_3 = nn.Linear(self.unit*2, self.unit)
        self.fc2   = nn.Linear(self.unit, self.LD)
        self.fc3   = nn.Linear(self.LD, self.unit)
        self.fc4_1 = nn.Linear(self.unit, self.unit*2)
        self.fc4_2 = nn.Linear(self.unit*2, self.unit*3)
        self.fc4_3 = nn.Linear(self.unit*3, self.IA)

    def encoder(self, x):
        x = self.fc1_1(x)
        x = torch.selu(x)
        x = self.fc1_2(x)
        x = torch.selu(x)
        x = self.fc1_3(x)
        x = torch.selu(x)
        return x

    def full_connection(self, z):
        z = self.fc2(z)
        return z

    def tr_full_connection(self, y):
        y = torch.selu(y)
        y = self.fc3(y)
        y = torch.selu(y)
        return y

    def decoder(self, y):
        y = self.fc4_1(y)
        y = torch.selu(y)
        y = self.fc4_2(y)
        y = torch.selu(y)
        y = self.fc4_3(y)
        y = torch.sigmoid(y)###tanh
        return y
    
    def forward(self, x):
        if shape_log == True: print(f'x:{x.shape}')
        x = x.view(-1, self.IA)
        z = self.encoder(x).cuda()
        if shape_log == True: print(f'z = encorder(x):{z.shape}')
        z = z.view(self.BS, -1)
        z = self.full_connection(z).cuda()
        if shape_log == True: print(f'full_connection(z):{z.shape}')
        #----------------------------------------
        y = self.tr_full_connection(z).cuda()
        if shape_log == True: print(f'tr_full_connection(z):{y.shape}')
        y = self.decoder(y).view(self.BS, self.IA).cuda()
        if shape_log == True: print(f'y:{y.shape}')
        y = y.view(self.BS, self.IA)##
        #lat_repr = z.view(self.BS, self.LD).cuda()
        return z, y#y, x, z, lat_repr
