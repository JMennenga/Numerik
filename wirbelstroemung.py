import numpy as np
import scipy.sparse as sparse
import scipy.sparse.linalg as splinalg
from stencil import Ableitung
import matplotlib.pyplot as plt

class Wirbelstroemung:
    def __init__(self, options):
        #Übernehmen des Options-Dict als klassen-Attribute
        self.image = options['image']
        self.h = options['h']
        self.CFL = options['CFL']
        self.kin_vis = options['kin_vis']
        self.inverted = options['inverted']
        self.w0string = options['text']
        self.maxOrdnung = options['order']

        # maxOrdnung = 0 bzw. 1 soll first order heißen
        if self.maxOrdnung in [0,1]:
            self.maxOrdnung = 2

        #auslesen der Grid-Größe
        self.gridshape = self.image[:, :, 1].shape
        self.gridlength = self.gridshape[0]*self.gridshape[1]

        #auslesen des vorgegebenen Randes (skaliert)
        self.psi_rand = np.array(self.image[:, :, 0]).reshape(self.gridlength)
        self.u_rand = ((np.array(self.image[:, :, 1]).reshape(self.gridlength)*0xFF) - 0x7F) / 0x9F
        self.v_rand = ((np.array(self.image[:, :, 2]).reshape(self.gridlength)*0xFF) - 0x7F) / 0x9F
        self.b_rand = np.array(self.image[:, :, 3]).reshape(self.gridlength)
        self.b_rand = np.array(self.b_rand == 1)

        #Randorientierung berechnen
        self.dir_rand = self.getdirection(self.b_rand.reshape(self.gridshape))





    def getdirection(self, b_rand):
        """
        Findet die Randorientierung von jedem Rand-Feld
        Dabei gilt:
            0 -- kein rand
            1 -- Rand ist pos-y-Richtung (S) orientiert d.h. 'Wasser' in neg-y-Richtung
            2 -- Rand ist neg-y-Richtung (N) orientiert
            4 -- Rand ist pos-x-Richtung (O) orientiert
            8 -- Rand ist neg-x-Richtung (W) orientiert
        Sonstige werte sind superpositionen dieser Fälle

        Parameter:
            b_rand: numpy ndarray mit 'shape' (m,n)

        Returns:
            dir_rand: numpy ndarray mit 'shape' (m,n)
        """
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

    def get_w0(self):
        #Anfangsbedingung auf [0,1],[0,1] skaliert
        xx = np.linspace(0, 1, self.gridshape[1])
        yy = np.linspace(0, 1, self.gridshape[0])
        XX, YY = np.meshgrid(xx, yy)

        # nicht schön -- user controlled eval
        # Sicherheitsrisiko
        return eval(self.w0string).reshape(self.gridlength)

    def setup(self):

        # Anlegen aller benötigten Ableitungen
        self.D1x = Ableitung(self.gridshape, self.h, 0, 1, 0, self.maxOrdnung, rand=self.b_rand)
        self.D1y = Ableitung(self.gridshape, self.h, 1, 1, 0, self.maxOrdnung, rand=self.b_rand)

        self.D1o = Ableitung(self.gridshape, self.h, 0, 1,  0.5, self.maxOrdnung, rand=self.b_rand)
        self.D1w = Ableitung(self.gridshape, self.h, 0, 1, -0.5, self.maxOrdnung, rand=self.b_rand)
        self.D1s = Ableitung(self.gridshape, self.h, 1, 1,  0.5, self.maxOrdnung, rand=self.b_rand)
        self.D1n = Ableitung(self.gridshape, self.h, 1, 1, -0.5, self.maxOrdnung, rand=self.b_rand)

        self.Lap0 = Ableitung(self.gridshape, self.h, 0, 2, 0, self.maxOrdnung, rand=self.b_rand)
        self.Lap0 = self.Lap0.add(Ableitung(self.gridshape, self.h, 1, 2,
                                            0, self.maxOrdnung, rand=self.b_rand))

        #Modifizieren der Rand-Laplace Zeilen (löschen)
        self.Lap0.randmod(self.b_rand, 'r')

        #Diagonalelemente einfügen
        self.Lap1 = self.Lap0.copy()
        self.Lap1.randmod(self.b_rand, 'd1')

        # W-Rand Matrix Erstellen von listen mit Rand-Matrix-Diagonalelementen (Wasser einseitig)
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

        # W-Rand Matrix Erstellen von listen mit Rand-Matrix-Diagonalelementen
        # (Wasser auf 2 seiten (Ecke))
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

        [self.b_rand, self.dir_rand] = [a.reshape(self.gridlength) for a in [self.b_rand, self.dir_rand]]


        # Auswählen der benötigten Listenelemente anhand der Randorientierung
        # (self.dir_rand == orientation des Stencils) -> boolean vektor
        # boolean Diagonalmatrix wird dann mit Ableitungsmatrix multipliziert
        # Addition der Einzelorientationsmatrizen
        self.Lap_rand = sparse.csr_matrix((self.gridlength, self.gridlength))
        self.Lap_rand = (1/self.h**2) * (sparse.diags(self.dir_rand == 0x1, dtype=bool)*b1[0]  # S-Rand
                                         + sparse.diags(self.dir_rand == 0x2, dtype=bool)*b1[2]  # N-Rand
                                         + sparse.diags(self.dir_rand == 0x4, dtype=bool)*b1[1]  # O-Rand
                                         + sparse.diags(self.dir_rand == 0x8, dtype=bool) * \
                                         b1[3]  # W-Rand  (heh SNOW...)

                                         + sparse.diags(self.dir_rand == 0x5, dtype=bool)*b2[0]  # SO
                                         + sparse.diags(self.dir_rand == 0x6, dtype=bool)*b2[1]  # NO
                                         + sparse.diags(self.dir_rand == 0x9, dtype=bool)*b2[3]  # SW
                                         + sparse.diags(self.dir_rand == 0xA, dtype=bool)*b2[2]  # SO
                                         )

        del b1
        del b2


        #Erstellen der 'Ghost-Point korrektur - Matrix'
        #x beschreibt in diesem fall die Flussrichtung
        self.Neumann_Korrekturx = 2 / self.h * (sparse.csr_matrix(sparse.diags(self.dir_rand & 0x1, dtype=bool), dtype = float)
                                                - sparse.csr_matrix(sparse.diags(self.dir_rand & 0x2, dtype=bool), dtype = float))

        self.Neumann_Korrektury = 2 / self.h * (sparse.csr_matrix(sparse.diags(self.dir_rand & 0x4, dtype=bool), dtype = float)
                                                - sparse.csr_matrix(sparse.diags(self.dir_rand & 0x8, dtype=bool), dtype = float))

        #Konvertieren von sparse-Aufbauformat auf effizienteres Rechenformat
        self.Lap_rand = sparse.csr_matrix(self.Lap_rand)

        [self.D1x, self.D1y, self.D1w, self.D1o, self.D1s, self.D1n, self.Lap0, self.Lap1] \
        = [i.final() for i in [self.D1x, self.D1y, self.D1w, self.D1o, self.D1s, self.D1n, self.Lap0, self.Lap1]]

        self.invert()

    def invert(self):
        if self.inverted:
            self.Lap1i = splinalg.inv(self.Lap1)

    def rk4(self, stopevent, f, ww):
        """
        Standard rk4 - Generatorfunktion mit
        ww' = f(ww,t)
        Läuft als Endlosschleife bis stopevent gesetzt wird

        Parameter:
            f : DGL - Funktion f(ww,t)
            ww : Anfangsbedingung ndarray
            stopevent: Event
        """
        t = 0
        dt = 0  #im ersten Durchlauf wird der erste zeitschritt bestimmt

        while not stopevent.is_set():
            [k1, ret_u, ret_v, draw_ww] = f(ww, t)
            k2 = f(ww + k1*dt/2, t+dt/2)[0]
            k3 = f(ww + k2*dt/2, t+dt/2)[0]
            [k4, u, v] = f(ww + k3*dt, t+dt)[:3]

            ww += dt*(k1 + 2*k2 + 2*k3 + k4)/6
            t += dt

            # Schrittweitenmodulation
            dt = self.h * self.CFL/max(np.abs(np.append(u, v)))
            yield [t, draw_ww]

        #stopevent.clear()
        yield [t, draw_ww]

    def rhs(self, w, t):

        #Stepfunktion

        w[self.b_rand] = 0

        # psi-Integration
        if self.inverted:
            psi_innen = self.Lap1i * (w - self.Lap0 * self.psi_rand)
        else:
            psi_innen = splinalg.spsolve(self.Lap1, (w - self.Lap0 * self.psi_rand))

        psi_tot = self.psi_rand + psi_innen

        #Berechnung von u,v
        u = self.D1y * (psi_tot)
        v = -self.D1x * (psi_tot)

        #Ersetzen der u,v Randelemente
        u[np.array(self.dir_rand & 0x3, dtype=bool)
          ] = self.u_rand[np.array(self.dir_rand & 0x3, dtype=bool)]

        v[np.array(self.dir_rand & 0xC, dtype=bool)
          ] = self.v_rand[np.array(self.dir_rand & 0xC, dtype=bool)]

        #Berechen von ww_rand mittels der Rand-Matrizen
        ww_rand = self.Lap_rand * (self.psi_rand + psi_innen) + \
             self.Neumann_Korrekturx * u + self.Neumann_Korrektury * v


        #Upwind-Bedingung
        ax = u > 0
        ay = v > 0
        # n = np.argmax(u[self.b_rand])
        # print('u_max : '+ str(u[self.b_rand][n]))#n%self.gridshape[0]) + ', ' + str(n/self.gridshape[0]))
        # n = np.argmax(v[self.b_rand])
        # print('v_max : '+ str(v[self.b_rand][n]))#n%self.gridshape[0]) + ', ' + str(n/self.gridshape[0]))

        self.ww = w + ww_rand

        rhs = -((np.multiply(ax, self.D1w * (u * (w + ww_rand))))
                + (np.multiply(np.logical_not(ax), self.D1o * (u * (w + ww_rand))))
                + (np.multiply(ay, self.D1n * (v * (w + ww_rand))))
                + (np.multiply(np.logical_not(ay), self.D1s * (v * (w + ww_rand))))

                - self.kin_vis * (self.Lap0 * (w + ww_rand))    #Reibungsterm
                )
                
        return [rhs, u, v, self.ww]
