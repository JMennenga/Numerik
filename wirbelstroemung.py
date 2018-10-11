import numpy as np
import scipy.sparse as sparse
import scipy.sparse.linalg as splinalg
import matplotlib.pyplot as plt
from stencil import Ableitung


class Wirbelstroemung:
    def __init__(self, path):

        self.path = path
        self.h = 0.1
        self.CFL = 0.9
        self.kin_vis = 0.1
        self.inverted = False
        self.w0string = ""
        self.maxOrdnung = 10

        image = np.array(plt.imread(path))
        print(image.shape)
        self.gridshape = image[:, :, 1].shape
        self.gridlength = self.gridshape[0]*self.gridshape[1]

        self.psi_rand = np.array(image[:, :, 0]).reshape(self.gridlength)
        self.u_rand = ((np.array(image[:, :, 1]).reshape(self.gridlength)*0xFF) - 0x7F) / 0x9F
        self.v_rand = ((np.array(image[:, :, 2]).reshape(self.gridlength)*0xFF) - 0x7F) / 0x9F
        self.b_rand = np.array(image[:, :, 3], dtype=bool).reshape(self.gridlength)

        self.dir_rand = self.getdirection(self.b_rand.reshape(self.gridshape))

    def getdirection(self, b_rand):

        dir_rand = np.zeros(self.gridshape, dtype=int)

        for i, value in np.ndenumerate(b_rand):
            if value:

                # Rand in Pos-y Richtung
                if i[0] != 0:
                    if not b_rand[i[0]-1, i[1]]:
                        dir_rand[i] = + 1

                # Rand in Neg-y Richtung
                if i[0] != self.gridshape[0]-1:
                    if not b_rand[i[0]+1, i[1]]:
                        dir_rand[i] = + 2

                # Rand in Pos-x Richtung
                if i[1] != 0:
                    if not b_rand[i[0], i[1]-1]:
                        dir_rand[i] = + 4

                # Rand in Neg-x Richtung
                if i[1] != self.gridshape[1]-1:
                    if not b_rand[i[0], i[1]+1]:
                        dir_rand[i] = + 8

        return dir_rand

    def get_w0():
        xx = np.linspace(0, 1, self.gridshape[0])
        yy = np.linspace(0, 1, self.gridshape[1])
        XX, YY = np.meshgrid(xx, yy)
        return eval(self.w0string)

    def setup(self):
        self.D1x = Ableitung(self.gridshape, self.h, 0, 1, 0, self.maxOrdnung, rand=self.b_rand)
        self.D1y = Ableitung(self.gridshape, self.h, 1, 1, 0, self.maxOrdnung, rand=self.b_rand)

        self.D1o = Ableitung(self.gridshape, self.h, 0, 1,  0.5, self.maxOrdnung-1, rand=self.b_rand)
        self.D1w = Ableitung(self.gridshape, self.h, 0, 1, -0.5, self.maxOrdnung-1, rand=self.b_rand)
        self.D1s = Ableitung(self.gridshape, self.h, 1, 1,  0.5, self.maxOrdnung-1, rand=self.b_rand)
        self.D1n = Ableitung(self.gridshape, self.h, 1, 1, -0.5, self.maxOrdnung-1, rand=self.b_rand)

        self.Lap0 = Ableitung(self.gridshape, self.h, 0, 2, 0, self.maxOrdnung+1, rand=self.b_rand)
        self.Lap0 = self.Lap0.add(Ableitung(self.gridshape, self.h, 1, 2,
                                            0, self.maxOrdnung+1, rand=self.b_rand))

        self.Lap0.randmod(self.b_rand, 'r')

        self.Lap1 = self.Lap0.copy()
        self.Lap1.randmod(self.b_rand, 'd1')

        # w-Rand Matrix
        a = np.array([2, 1, 1, -4])
        a = (a * np.ones((self.gridlength, 4))).transpose()
        b1 = []
        b1.append(sparse.spdiags(a, [-self.gridshape[1], -1, 1, 0],
                                 self.gridlength, self.gridlength))
        b1.append(sparse.spdiags(a, [-1, self.gridshape[1], -
                                     self.gridshape[1], 0], self.gridlength, self.gridlength))
        b1.append(sparse.spdiags(a, [self.gridshape[1], 1, -1, 0],
                                 self.gridlength, self.gridlength))
        b1.append(sparse.spdiags(
            a, [1, -self.gridshape[1], self.gridshape[1], 0], self.gridlength, self.gridlength))

        a = np.array([1.5, 1.5, 0.5, 0.5, -4])
        a = (a * np.ones((self.gridlength, 5))).transpose()
        b2 = []
        b2.append(sparse.spdiags(
            a, [-self.gridshape[1], -1, self.gridshape[1], 1, 0], self.gridlength, self.gridlength))
        b2.append(sparse.spdiags(
            a, [-1, self.gridshape[1], 1, -self.gridshape[1], 0], self.gridlength, self.gridlength))
        b2.append(sparse.spdiags(
            a, [self.gridshape[1], 1, -self.gridshape[1], -1, 0], self.gridlength, self.gridlength))
        b2.append(sparse.spdiags(a, [1, -self.gridshape[1], -1,
                                     self.gridshape[1], 0], self.gridlength, self.gridlength))

        [b_rand, dir_rand] = [a.reshape(self.gridlength) for a in [self.b_rand, self.dir_rand]]

        self.Lap_rand = sparse.csr_matrix((self.gridlength, self.gridlength))
        self.Lap_rand = (1/self.h**2) * (sparse.diags(dir_rand == 0x1, dtype=bool)*b1[0]  # S-Rand
                                         + sparse.diags(dir_rand == 0x2, dtype=bool)*b1[2]  # N-Rand
                                         + sparse.diags(dir_rand == 0x4, dtype=bool)*b1[1]  # O-Rand
                                         + sparse.diags(dir_rand == 0x8, dtype=bool) * \
                                         b1[3]  # W-Rand  (heh SNOW...)

                                         + sparse.diags(dir_rand == 0x5, dtype=bool)*b2[0]  # SO
                                         + sparse.diags(dir_rand == 0x6, dtype=bool)*b2[1]  # NO
                                         + sparse.diags(dir_rand == 0x9, dtype=bool)*b2[3]  # SW
                                         + sparse.diags(dir_rand == 0xA, dtype=bool)*b2[2]  # SO
                                         )

        self.Neumann_Korrekturx = 2 * self.h * (sparse.diags(dir_rand & 0x1, dtype=float)
                                                - sparse.diags(dir_rand & 0x2, dtype=float))

        self.Neumann_Korrektury = 2 * self.h * (sparse.diags(dir_rand & 0x4, dtype=float)
                                                - sparse.diags(dir_rand & 0x8, dtype=float))

        self.Lap_rand = sparse.csr_matrix(self.Lap_rand)

        for i in [self.D1x, self.D1y, self.D1w, self.D1o, self.D1s, self.D1n, self.Lap0, self.Lap1]:
            i.final()

    def invert(self):
        if self.inverted:
            self.Lap1i = splinalg.inv(self.Lap1)

    def rk4(self, stopevent, f, ww):

        t = 0
        dt = 0

        while not stopevent.is_set():
            [k1, psi_tot, ret_u, ret_v, draw_ww] = f(ww, t)
            k2 = f(ww + k1*dt/2, t+dt/2)[0]
            k3 = f(ww + k2*dt/2, t+dt/2)[0]
            [k4, u, v] = f(ww + k3*dt, t+dt)[0, 2, 3]

            ww += dt*(k1 + 2*k2 + 2*k3 + k4)/6
            t += dt

            # Schrittweitenmodulation
            dt = self.h * self.CFL/max(np.abs(np.append(u, v)))

            yield [ww, psi_tot, ret_u, ret_v, draw_ww]
        else:
            stopevent.clear()
            yield [ww, psi_tot, ret_u, ret_v, draw_ww]

    def rhs(self, w, t):
        w[self.b_rand] = 0

        if self.inverted:
            psi_innen = self.Lap1i * (w - self.Lap0 * self.psi_rand)
        else:
            psi_innen = splinalg.spsolve(self.Lap1, (w - self.Lap0.matrix * self.psi_rand))

        psi_tot = self.psi_rand + psi_innen

        u = self.D1y * (psi_tot)
        v = -self.D1x * (psi_tot)

        u[np.array(self.dir_rand & 0x3, dtype=bool)
          ] = self.u_rand[np.array(self.dir_rand & 0x3, dtype=bool)]

        v[np.array(self.dir_rand & 0xC, dtype=bool)
          ] = self.v_rand[np.array(self.dir_rand & 0xC, dtype=bool)]

        ww_rand = self.Lap_rand * (self.psi_rand + psi_innen) + \
            self.Neumann_Korrekturx * u + self.Neumann_Korrekturx * v

        ax = u > 0
        ay = v > 0

        self.ww = w + ww_rand

        rhs = -((np.multiply(ax, self.D1w(u * (w + ww_rand))))
                + (np.multiply(np.logical_not(ax), self.D1o(u * (w + ww_rand))))
                + (np.multiply(ay, self.D1n(v * (w + ww_rand))))
                + (np.multiply(np.logical_not(ay), self.D1s(v * (w + ww_rand))))

                - self.kin_vis * (self.Lap0 * (w + ww_rand))
                )
        return [rhs, psi_tot, u, v, self.ww]
