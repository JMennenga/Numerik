import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from stencil import Stencil, Ableitung

# i = 4
# a = np.ceil(np.arange(i)- np.floor(i/2)-0.5)
# print(a)
b_rand = np.zeros((10,10), dtype=bool)
b_rand[0,:] = 1
b_rand[9,:] = 1
b_rand[:,0] = 1
b_rand[:,9] = 1
a = Ableitung((10,10),0,1,0,10,b_rand)


plt.imshow(a.matrix.todense())
plt.colorbar()
plt.show()
