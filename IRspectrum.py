# -*- coding: utf-8 -*-
"""
Created on Sat Jul 16 12:57:57 2022

@author: nan-n
"""
from rdkit import rdBase, Chem
from rdkit.Chem import AllChem, Draw
from rdkit.Chem.Draw import rdMolDraw2D
from IPython.display import SVG
from matplotlib.colors import ColorConverter
import numpy as np
import pandas as pd
import math
from scipy.optimize import curve_fit
from scipy.stats import cauchy, norm
import seaborn as sns
from matplotlib import pyplot as plt
from scipy.stats import cauchy, norm
from IPython.display import Image, display


#データの準備
df = pd.read_csv('all_sdf.csv', index_col=0)
df2 = pd.read_csv('qm9.csv', index_col=0)

#化合物
molenum = 11999

samplesmiles = df2['smiles'][molenum]
molecule = Chem.MolFromSmiles(samplesmiles)
Draw.MolToFile(molecule, 'molecule.png')
t = df.iloc[molenum].to_list()
mole_freq = [x for x in t if math.isnan(x) == False]

sns.set(style="white")

x = np.linspace(0, 4000, 10000)
y = np.zeros(10000)
for p in mole_freq:
    X = cauchy(loc=p, scale=10)
    y1 = X.pdf(x)
    y += y1

fig, ax = plt.subplots()
ax.plot(x, y)
ax.grid()
ax.invert_xaxis()
ax.invert_yaxis()

plt.show()
display(Image(filename="molecule.png"))
