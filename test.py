#!/usr/bin/env python
import os
import glob
import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sparse
import scipy.sparse.linalg as splinalg
import matplotlib.colors as colors
from stencil import Ableitung
#from stencil_orig import Ableitung_orig, Stencil


def sim():
    def rhs(w, t):
        w[b_rand] = 0
        #psi_innen = splinalg.spsolve(Lap1.matrix, (w - Lap0.matrix * psi_rand))
        psi_innen = Lap1i*(w - Lap0.matrix * psi_rand)
        u = D1y.matrix * (psi_rand + psi_innen)
        u[np.array(dir_rand & 0x3, dtype=bool)
          ] = u_rand[np.array(dir_rand & 0x3, dtype=bool)]

        v = -D1x.matrix * (psi_rand + psi_innen)
        v[np.array(dir_rand & 0xC, dtype=bool)
          ] = v_rand[np.array(dir_rand & 0xC, dtype=bool)]

        ww_rand = Lap_rand * (psi_rand + psi_innen) + \
            Neumann_Korrekturx * u + Neumann_Korrekturx * v

        ax = u > 0
        ay = v > 0

        rhs = -((np.multiply(ax, D1w.dot(u * (w + ww_rand))))
                + (np.multiply(np.logical_not(ax), D1o.dot(u * (w + ww_rand))))
                + (np.multiply(ay, D1n.dot(v * (w + ww_rand))))
                + (np.multiply(np.logical_not(ay), D1s.dot(v * (w + ww_rand))))

                - kin_vis * (Lap0.matrix * (w + ww_rand))
                )
        return [rhs, u, v]

    def rk4(f, ww, max_iter):

        t = 0
        dt = 0

        for i in range(max_iter + 1):
            k1 = f(ww, t+dt/2)[0]
            k2 = f(ww + k1*dt/2, t+dt/2)[0]
            k3 = f(ww + k2*dt/2, t+dt/2)[0]
            [k4, u, v] = f(ww + k3*dt, t+dt)

            ww += dt*(k1 + 2*k2 + 2*k3 + k4)/6
            t += dt

            # Schrittweitenmodulation
            dt = h * CFL/max(np.abs(np.append(u, v)))

            yield ww

    def getdirection(b_rand):

        dir_rand = np.zeros(gridshape, dtype=int)

        for i, value in np.ndenumerate(b_rand):
            if value:

                # Rand in Pos-y Richtung
                if i[0] != 0:
                    if not b_rand[i[0]-1, i[1]]:
                        dir_rand[i] = + 1

                # Rand in Neg-y Richtung
                if i[0] != gridshape[0]-1:
                    if not b_rand[i[0]+1, i[1]]:
                        dir_rand[i] = + 2

                # Rand in Pos-x Richtung
                if i[1] != 0:
                    if not b_rand[i[0], i[1]-1]:
                        dir_rand[i] = + 4

                # Rand in Neg-x Richtung
                if i[1] != gridshape[1]-1:
                    if not b_rand[i[0], i[1]+1]:
                        dir_rand[i] = + 8

        return dir_rand

    # Konstantendef
    global simloop_active
    kin_vis = 0.1
    h = 1
    CFL = 0.8

    dir = os.path.dirname(__file__)
    paths = glob.glob(dir + "\\*.png")
    loop_count = 0
    for image_path in paths:
        print(str(loop_count) + ': ' + image_path)
        loop_count += 1

        image_path = paths[2]  # paths[int(input())]
        image = np.array(plt.imread(image_path))

    gridshape = image[:, :, 1].shape
    gridlength = gridshape[0]*gridshape[1]

    psi_rand = np.array(image[:, :, 0]).reshape(gridlength)
    u_rand = ((np.array(image[:, :, 1]).reshape(gridlength)*0xFF) - 0x7F) / 0x9F
    v_rand = ((np.array(image[:, :, 2]).reshape(gridlength)*0xFF) - 0x7F) / 0x9F
    b_rand = np.array(image[:, :, 3], dtype=bool)

    print(v_rand)

    dir_rand = getdirection(b_rand)

    xx = np.linspace(0, 1, gridshape[0])
    yy = np.linspace(0, 1, gridshape[1])
    XX, YY = np.meshgrid(xx, yy)

    WW = np.sin(3 * np.pi * (XX)) * np.sin(4 * np.pi * YY)
    ww = WW.reshape(gridlength)

    D1x = Ableitung(gridshape, h, 0, 1, 0, 10, rand=b_rand)
    D1y = Ableitung(gridshape, h, 1, 1, 0, 10, rand=b_rand)

    D1o = Ableitung(gridshape, h, 0, 1,  0.5, 9, rand=b_rand)
    D1w = Ableitung(gridshape, h, 0, 1, -0.5, 9, rand=b_rand)
    D1s = Ableitung(gridshape, h, 1, 1,  0.5, 9, rand=b_rand)
    D1n = Ableitung(gridshape, h, 1, 1, -0.5, 9, rand=b_rand)

    Lap0 = Ableitung(gridshape, h, 0, 2, 0, 10, rand=b_rand)
    Lap0 = Lap0.add(Ableitung(gridshape, h, 1, 2, 0, 10, rand=b_rand))

    Lap0.randmod(b_rand, 'r')

    Lap1 = Lap0.copy()
    Lap1.randmod(b_rand, 'd1')

    # Rand w-Matrix
    a = np.array([2, 1, 1, -4])
    a = (a * np.ones((gridlength, 4))).transpose()
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

    [b_rand, dir_rand] = [a.reshape(gridlength) for a in [b_rand, dir_rand]]

    Lap_rand = sparse.csr_matrix((gridlength, gridlength))
    Lap_rand = (sparse.diags(dir_rand == 0x1, dtype=bool)*b1[0]  # S-Rand
                + sparse.diags(dir_rand == 0x2, dtype=bool)*b1[2]  # N-Rand
                + sparse.diags(dir_rand == 0x4, dtype=bool)*b1[1]  # O-Rand
                + sparse.diags(dir_rand == 0x8, dtype=bool) * b1[3]  # W-Rand  (heh SNOW...)

                + sparse.diags(dir_rand == 0x5, dtype=bool)*b2[0]  # SO
                + sparse.diags(dir_rand == 0x6, dtype=bool)*b2[1]  # NO
                + sparse.diags(dir_rand == 0x9, dtype=bool)*b2[3]  # SW
                + sparse.diags(dir_rand == 0xA, dtype=bool)*b2[2]  # SO
                )

    Neumann_Korrekturx = 2 * (sparse.diags(dir_rand & 0x1, dtype=float)
                              - sparse.diags(dir_rand & 0x2, dtype=float))

    Neumann_Korrektury = 2 * (sparse.diags(dir_rand & 0x4, dtype=float)
                              - sparse.diags(dir_rand & 0x8, dtype=float))

    del b1
    del b2

    Lap_rand = sparse.csr_matrix(Lap_rand)

    for i in [D1x, D1y, D1w, D1o, D1s, D1n, Lap0, Lap1]:
        i.final()
    Lap1i = splinalg.inv(Lap1.matrix)

    global drawobj

    simloop_active = True
    # Thread.wait()
    for ww in rk4(rhs, ww, 1000):
        drawobj = ww

    simloop_active = False
