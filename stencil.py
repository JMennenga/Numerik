import numpy as np
import math
import scipy.sparse as sparse
import copy

import matplotlib.pyplot as plt

class Stencil:
    def __init__(self, ord, s):
        self.ordnung = ord
        self.pos = np.array(s)
        self.coefficients = self.getcoefficients(self.pos, ord)
        self.length = len(s)

    def getcoefficients(self, s, ord):
        vand = np.zeros((len(s), len(s)))
        for i in range(0, len(s)):

            vand[i, :] = s ** i         # 0^0 wird hier als 1 betrachtet

        b = np.zeros((len(s), 1))
        b[ord] = math.factorial(ord)

        c = np.linalg.solve(vand, b)
        return c

    def len(self):
        return self.length

class Ableitung:

    def __init__(self, shape, orientation, ablOrdnung = 1,  offset =0, maxOrdnung = 2, rand = []):

        self.gridshape = shape
        self.sidelength = shape[0]*shape[1]

        if rand != []:
            rand_dist = np.zeros(shape)
            for index, value in np.ndenumerate(rand):
                i = 0
                while not 1 in (rand[index[0]-i,index[1]],
                                rand[index[0]+i,index[1]],
                                rand[index[0],index[1]-i],
                                rand[index[0],index[1]+i]):
                    i += 1

                rand_dist[index] = i

            plt.imshow(rand_dist)
            plt.show()
            if (offset%1):
                min = 2
            else:
                min = 1

            rand_dist = rand_dist.reshape(self.sidelength)

            self.matrix = sparse.lil_matrix((self.sidelength,self.sidelength))
            n = 1
            for i in range(min,maxOrdnung-1,2):
                a = np.ceil(np.arange(i+1)- np.floor((i+1)/2) + offset)
                s = Stencil(ablOrdnung, a)
                print(a)
                self.matrix = self.matrix + (sparse.diags(rand_dist == np.ceil(n+abs(offset)), dtype = bool) * self.sten2mat(s, orientation))
                n += 1

            a = np.ceil(np.arange(maxOrdnung+1)- np.floor((maxOrdnung+1)/2) + offset)
            s = Stencil(ablOrdnung, a)
            print(a)
            self.matrix = self.matrix + (sparse.diags(rand_dist >= n, dtype = bool) * self.sten2mat(s, orientation))


        else:
            self.gridshape = shape
            self.sidelength = shape[0]*shape[1]
            if (offset%0):
                maxOrdnung += 1
            stencil = Stencil(ablOrdnung, np.ceil(np.arange(maxOrdnung)- np.floor(maxOrdnung/2) + offset))
            self.matrix = self.sten2mat(stencil, Orientation)
            #self.matrix.eliminate_zeros()

    def copy(self):
        return copy.deepcopy(self)

    def sten2mat(self, stencil, Orientation, *args):

        matrix = sparse.lil_matrix((self.sidelength,self.sidelength))

        if (Orientation == 0):
            a = stencil.coefficients * np.ones((stencil.len(),  self.gridshape[0]))
            abl1D = sparse.spdiags(a, stencil.pos, self.gridshape[0], self.gridshape[0])
            matrix = sparse.kron(sparse.eye(self.gridshape[1]), abl1D, 'lil')
        elif (Orientation == 1):
            a = stencil.coefficients * np.ones((stencil.len(),  self.gridshape[1]))
            abl1D = sparse.spdiags(a, stencil.pos, self.gridshape[1], self.gridshape[1])
            matrix = sparse.kron(abl1D, sparse.eye(self.gridshape[0]), 'lil')

        return matrix

    def randmod(self, b_rand, modifytype = 'r'):
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

        if modifytype in ['row','r']:
            self.matrix[b_rand,:] = np.zeros((true_sum,self.sidelength))

        if modifytype in ['column','c']:
            self.matrix[:,b_rand] = np.zeros((self.sidelength,true_sum))

        if modifytype in ['diag_one','d1']:
            d = self.matrix.diagonal()
            d[b_rand] = 1           #np.ones(true_sum)
            self.matrix.setdiag(d)

        #self.matrix.eliminate_zeros()

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

    def dot(self, other):
        return self.matrix.dot(other)
