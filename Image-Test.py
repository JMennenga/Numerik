import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from stencil import Stencil, Ableitung

# i = 4
# a = np.ceil(np.arange(i)- np.floor(i/2)-0.5)
# print(a)

b_rand = np.zeros((100, 100), dtype=bool)
b_rand[0, :] = 1
b_rand[99, :] = 1
b_rand[:, 0] = 1
b_rand[:, 99] = 1
a = Ableitung((100, 100), 0, 1, 0, 9, b_rand)

plt.imshow(a.matrix.todense())
plt.colorbar()
plt.show()
