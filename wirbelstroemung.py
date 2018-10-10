

class Wirbelstroemung:
    __init__(self, path):

        self.path = path
        self.h = 0.1
        self.CFL = 0.9
        self.kin_vis = 0.1

        self.image = np.array(plt.imread(path))

        self.gridshape = image[:, :, 1].shape
        self.gridlength = gridshape[0]*gridshape[1]

        self.psi_rand = np.array(image[:, :, 0]).reshape(gridlength)
        self.u_rand = ((np.array(image[:, :, 1]).reshape(gridlength)*0xFF) - 0x7F) / 0x9F
        self.v_rand = ((np.array(image[:, :, 2]).reshape(gridlength)*0xFF) - 0x7F) / 0x9F
        self.b_rand = np.array(image[:, :, 3], dtype=bool)

        self.dir_rand = getdirection(b_rand)

        xx = np.linspace(0, 1, gridshape[0])
        yy = np.linspace(0, 1, gridshape[1])
        XX, YY = np.meshgrid(xx, yy)

    def getdirection(b_rand):

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

    def setup(self,maxOrdnung):
        self.D1x = Ableitung(self.gridshape, h, 0, 1, 0, maxOrdnung, rand = self.b_rand)
        self.D1y = Ableitung(self.gridshape, h, 1, 1, 0, maxOrdnung, rand = self.b_rand)

        self.D1o = Ableitung(self.gridshape, h, 0, 1,  0.5, maxOrdnung, rand = self.b_rand)
        self.D1w = Ableitung(self.gridshape, h, 0, 1, -0.5, maxOrdnung, rand = self.b_rand)
        self.D1s = Ableitung(self.gridshape, h, 1, 1,  0.5, maxOrdnung, rand = self.b_rand)
        self.D1n = Ableitung(self.gridshape, h, 1, 1, -0.5, maxOrdnung, rand = self.b_rand)

        self.Lap0 = Ableitung(self.gridshape, h, 0, 2, 0, maxOrdnung, rand = self.b_rand)
        self.Lap0 = self.Lap0.add(Ableitung(self.gridshape, h, 1, 2, 0, maxOrdnung, rand = self.b_rand))

        self.Lap0.randmod(self.b_rand, 'r')

        self.Lap1 = Lap0.copy()
        self.Lap1.randmod(self.b_rand, 'd1')

        # w-Rand Matrix
        a = np.array([2, 1, 1, -4])
        a = (a * np.ones((self.gridlength, 4))).transpose()
        b1 = []
        b1.append(sparse.spdiags(a, [-gridshape[1], -1, 1, 0], gridlength, gridlength))
        b1.append(sparse.spdiags(a, [-1, gridshape[1], -
                                     gridshape[1], 0], gridlength, gridlength))
        b1.append(sparse.spdiags(a, [gridshape[1], 1, -1, 0], gridlength, gridlength))
        b1.append(sparse.spdiags(
            a, [1, -gridshape[1], gridshape[1], 0], gridlength, gridlength))

        a = np.array([1.5, 1.5, 0.5, 0.5, -4])
        a = (a * np.ones((gridlength, 5))).transpose()
        b2 = []
        b2.append(sparse.spdiags(
            a, [-gridshape[1], -1, gridshape[1], 1, 0], gridlength, gridlength))
        b2.append(sparse.spdiags(
            a, [-1, gridshape[1], 1, -gridshape[1], 0], gridlength, gridlength))
        b2.append(sparse.spdiags(
            a, [gridshape[1], 1, -gridshape[1], -1, 0], gridlength, gridlength))
        b2.append(sparse.spdiags(a, [1, -gridshape[1], -1,
                                     gridshape[1], 0], gridlength, gridlength))

        [b_rand, dir_rand] = [a.reshape(self.gridlength) for a in [self.b_rand, self.dir_rand]]

        self.Lap_rand = sparse.csr_matrix((self.gridlength, self.gridlength))
        self.Lap_rand = (1/self.h**2) * (sparse.diags(dir_rand == 0x1, dtype=bool)*b1[0]  # S-Rand
                    + sparse.diags(dir_rand == 0x2, dtype=bool)*b1[2]  # N-Rand
                    + sparse.diags(dir_rand == 0x4, dtype=bool)*b1[1]  # O-Rand
                    + sparse.diags(dir_rand == 0x8, dtype=bool)*b1[3]  # W-Rand  (heh SNOW...)

                    + sparse.diags(dir_rand == 0x5, dtype=bool)*b2[0]  # SO
                    + sparse.diags(dir_rand == 0x6, dtype=bool)*b2[1]  # NO
                    + sparse.diags(dir_rand == 0x9, dtype=bool)*b2[3]  # SW
                    + sparse.diags(dir_rand == 0xA, dtype=bool)*b2[2]  # SO
                    )

        self.Neumann_Korrekturx = 2 * self.h * (sparse.diags(dir_rand & 0x1, dtype=float)
                                  - sparse.diags(dir_rand & 0x2, dtype=float))

        self.Neumann_Korrektury = 2 * self.h * (sparse.diags(dir_rand & 0x4, dtype=float)
                                  - sparse.diags(dir_rand & 0x8, dtype=float))


        self.Lap_rand = sparse.csr_matrix(Lap_rand)

        for i in [self.D1x,self.D1y,self.D1w,self.D1o,self.D1s,self.D1n,self.Lap0,self.Lap1]:
            i.final()

    def step(opt, args):
        def rk4(f, ww, max_iter):

            t = 0
            dt = 0

            for i in range(max_iter + 1):
                k1 = f(ww, t+dt/2)
                k2 = f(ww + k1*dt/2, t+dt/2)
                k3 = f(ww + k2*dt/2, t+dt/2)
                k4 = f(ww + k3*dt, t+dt)

                ww += dt*(k1 + 2*k2 + 2*k3 + k4)/6
                t += dt

                #Schrittweitenmodulation
                dt = h * CFL/max(np.abs(np.append(u, v)))

                yield ww

        def rhs(w, t):
            w[b_rand] = 0
            #psi_innen = splinalg.spsolve(Lap1.matrix, (w - Lap0.matrix * psi_rand))
            psi_innen = Lap1i*(w - Lap0.matrix * psi_rand)
            u = D1y.matrix * (psi_rand + psi_innen)
            u[np.array(self.dir_rand & 0x3, dtype=bool)
              ] = u_rand[np.array(self.dir_rand & 0x3, dtype=bool)]

            v = -D1x.matrix * (psi_rand + psi_innen)
            v[np.array(self.dir_rand & 0xC, dtype=bool)
              ] = v_rand[np.array(self.dir_rand & 0xC, dtype=bool)]

            ww_rand = Lap_rand * (psi_rand + psi_innen) + \
                Neumann_Korrekturx * u + Neumann_Korrekturx * v

            ax = u > 0
            ay = v > 0

            self.ww = w + ww_rand
            

            rhs = -((np.multiply(ax, D1w.dot(u * (w + ww_rand))))
                    + (np.multiply(np.logical_not(ax), D1o.dot(u * (w + ww_rand))))
                    + (np.multiply(ay, D1n.dot(v * (w + ww_rand))))
                    + (np.multiply(np.logical_not(ay), D1s.dot(v * (w + ww_rand))))

                    - kin_vis * (Lap0.matrix * (w + ww_rand))
                    )
            return rhs


        for w in rk4(rhs, ww, 10):
            drawobj = ww
