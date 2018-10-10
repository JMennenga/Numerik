import numpy as np
import math
import scipy.sparse as sparse
import copy

import matplotlib.pyplot as plt


class Stencil:
    def __init__(self, ord, h, s):
        self.ordnung = ord
        self.pos = np.array(s)
        self.coefficients = self.getcoefficients(self.pos, h, ord)
        self.length = len(s)

    def getcoefficients(self, s, h, ord):
        vand = np.zeros((len(s), len(s)))
        for i in range(0, len(s)):

            vand[i, :] = s ** i         # 0^0 wird hier als 1 betrachtet

        b = np.zeros((len(s), 1))
        b[ord] = math.factorial(ord)

        c = (h**(-ord)) * np.linalg.solve(vand, b)
        return c

    def len(self):
        return self.length


class Ableitung:

    def __init__(self, shape, h, orientation, ablOrdnung=1,  offset=0, maxOrdnung=10, rand=[]):
        """
            Patameter:
                shape (Tuple):
                    Gridshape this will affect
                orientation (int):
                    0 primary
                    1 secondary
                ablOrdnung:
                    Order of differentiation
                offset (int/2):
                    Offset of stencil-center compared to point computed
                maxOrdnung:
                    Highest order of Approximation used
                rand (bool numpy.ndarray):
                    true for boundary Points
        """

        self.gridshape = shape
        self.sidelength = shape[0]*shape[1]

        if rand.any():
            rand_dist = np.zeros(shape)
            for index, value in np.ndenumerate(rand):
                i = 0
                while True:
                    if orientation:
                        ls = [rand[index[0]-i, index[1]], rand[index[0]+i, index[1]]]
                    else:
                        ls = [rand[index[0], index[1]-i], rand[index[0], index[1]+i]]
                    if 1 in ls:
                        break
                    i += 1

                rand_dist[index] = i

            if not (offset % 1):                # Gilt nur eingeschränkt for hier verwendete Matrizen
                min = 2
            else:
                min = 1

            # plt.imshow(rand_dist)
            # plt.show()
            rand_dist = rand_dist.reshape(self.sidelength)

            self.matrix = sparse.lil_matrix((self.sidelength, self.sidelength))
            n = 1

            iter = list(range(min, maxOrdnung+1, 2))

            if len(iter) > 1:
                for i in iter:

                    a = np.ceil(np.arange(i+1) - ((i+1)/2) + offset)
                    s = Stencil(ablOrdnung, h, a)
                    if n == 1:
                        self.matrix = self.matrix + \
                            (sparse.diags(rand_dist <= n, dtype=bool) * self.sten2mat(s, orientation))

                    elif i == iter[-1]:
                        self.matrix = self.matrix + \
                            (sparse.diags(rand_dist >= n, dtype=bool) * self.sten2mat(s, orientation))

                    else:
                        self.matrix = self.matrix + \
                            (sparse.diags(rand_dist == n, dtype=bool) * self.sten2mat(s, orientation))

                    n += 1
            else:
                a = np.ceil(np.arange(iter[0]+1) - ((iter[0]+1)/2) + offset)
                s = Stencil(ablOrdnung, h, a)
                self.matrix = self.matrix + \
                    (self.sten2mat(s, orientation))

        else:
            self.gridshape = shape
            self.sidelength = shape[0]*shape[1]
            print(maxOrdnung)
            if not (offset % 1):
                maxOrdnung += 1
            print(maxOrdnung)
            stencil = Stencil(ablOrdnung, h, np.ceil(
                np.arange(maxOrdnung) - np.floor(maxOrdnung/2) + offset))
            self.matrix = self.sten2mat(stencil, orientation)
            # self.matrix.eliminate_zeros()

    def copy(self):
        return copy.deepcopy(self)

    def sten2mat(self, stencil, Orientation, *args):

        matrix = sparse.lil_matrix((self.sidelength, self.sidelength))

        if (Orientation == 0):
            a = stencil.coefficients * np.ones((stencil.len(),  self.gridshape[0]))
            abl1D = sparse.spdiags(a, stencil.pos, self.gridshape[0], self.gridshape[0])
            matrix = sparse.kron(sparse.eye(self.gridshape[1]), abl1D, 'lil')
        elif (Orientation == 1):
            a = stencil.coefficients * np.ones((stencil.len(),  self.gridshape[1]))
            abl1D = sparse.spdiags(a, stencil.pos, self.gridshape[1], self.gridshape[1])
            matrix = sparse.kron(abl1D, sparse.eye(self.gridshape[0]), 'lil')

        return matrix

    def randmod(self, b_rand, modifytype='r'):
        #
        # b_rand (m,n) or (m*n) numpy logical array
        # modifytype can be either
        # 'column' ('c')      replaces specified columns of matrix with 0
        # 'row' ('r')         replaces specified rows of matrix with 0
        # 'diag_one' ('d1')   replaces specified diagonal entries with 1
        #
        # defaults to 'r'
        b_rand = b_rand.reshape(self.sidelength)
        true_sum = np.sum(b_rand)

        if modifytype in ['row', 'r']:
            self.matrix[b_rand, :] = np.zeros((true_sum, self.sidelength))

        if modifytype in ['column', 'c']:
            self.matrix[:, b_rand] = np.zeros((self.sidelength, true_sum))

        if modifytype in ['diag_one', 'd1']:
            d = self.matrix.diagonal()
            d[b_rand] = 1  # np.ones(true_sum)
            self.matrix.setdiag(d)

        # self.matrix.eliminate_zeros()

    def todense(self):
        return self.matrix.todense()

    def add(self, other):
        res = copy.deepcopy(self)
        res.matrix = (self.matrix + other.matrix).tolil()
        return res

    def final(self):
        # csr and csc matrices are faster in arithmetic operations
        self.matrix = self.matrix.tocsr()
        self.matrix.eliminate_zeros()

        self = self.matrix


if __name__ == '__main__':

    b_rand = np.zeros((20,20), dtype=bool)
    b_rand[0,:] = 1
    b_rand[19,:] = 1
    b_rand[:,0] = 1
    b_rand[:,19] = 1
    A = Ableitung((20, 20), 1, 0, 1, 0, 4, b_rand)
    plt.imshow(A.matrix.todense())
    plt.show()

    A.final()

    plt.imshow(A.todense())
    plt.show()
