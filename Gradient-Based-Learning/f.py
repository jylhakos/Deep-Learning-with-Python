#$ pip3 install torch
import numpy as np
import torch
from d2l import torch as d2l
from mpl_toolkits import mplot3d

def f(x):
	return x * torch.cos(np.pi * x)

def annotate(text, xy, xytext):  #@save
	d2l.plt.gca().annotate(text, xy=xy, xytext=xytext, arrowprops=dict(arrowstyle='->'))

x = torch.arange(0.5, 1.5, 0.01)
d2l.set_figsize((4.5, 2.5))
d2l.plot(x, [f(x)], 'x', 'f(x)')
annotate('global minimum', (1.1, -1.05), (0.95, -0.5))
