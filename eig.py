import numpy as np
import stencil
import scipy.sparse as sparse
import scipy.linalg as lin
import matplotlib.pyplot as plt

# b_rand = np.zeros(21, dtype=bool)
# b_rand[0] = True
# b_rand[20] = True
#
# D1p = stencil.Ableitung((21, 1), 1, 1, offset=-0.5, maxOrdnung=1, rand=b_rand)
# D1n = stencil.Ableitung((21, 1), 1, 1, offset=-0.5, maxOrdnung=1, rand=b_rand)
# D1p.randmod(b_rand, 'r')
# D1p.randmod(b_rand, 'd1')
# D1n.randmod(b_rand, 'r')
# D1n.randmod(b_rand, 'd1')
# D1p = D1p.final().todense()
# D1n = D1n.final().todense()
#
#
# a = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], dtype=bool)
#
# M = np.diag(a) * D1n + np.diag(np.logical_not(a)) * D1p
# M = M
# print(M)

file = open('eigs.txt', 'r')
a = []
while True:
    item = file.readline().strip()
    for i in "()":
        item = item.replace(i, "")
    # print(item)
    if item == '':
        break
    a.append(complex(item))


# print(a)

re = np.real(a)
im = np.imag(a)
g0 = np.real(a) > 0
print(g0)

plt.scatter(re[g0],im[g0], c='C3')
plt.scatter(re[np.logical_not(g0)], im[np.logical_not(g0)], c = 'C0')
plt.title(r"Eigenwerte $\lambda$ von $L_{n}$", fontsize=20, fontweight=0, color='C0', loc='left', style='italic')
plt.title('CFL = '+ str(0.8), loc = 'right')
plt.ylabel(r"$\Im(\lambda$)")
plt.xlim(-1,0.5)
plt.ylim(-1,1)
plt.xlabel(r"$\Re(\lambda$)")
plt.show()
