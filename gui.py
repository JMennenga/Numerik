import matplotlib.pyplot as plt
import matplotlib.widgets as wdg
import numpy as np
fig = plt.figure(figsize=(12,9))

plt_frame = [0.15 ,0.15,0.8,0.8]
opt_frame =  [0.05,0.25, 0.1,0.2]
txt_frame =  [0.2, 0.05,0.7,0.05]

plt_ax = fig.add_subplot(111, position = plt_frame)
opt_ax = fig.add_axes(opt_frame, frameon = False)
txt_ax = fig.add_axes(txt_frame)

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
    
def submit(text):

    kin_vis = 0.01
    h = 0.5
    CFL = 0.8

    lock = threading.Lock()

    dir = os.path.dirname(__file__)
    paths = glob.glob(dir + "\\*.png")
    loop_count = 0
    for image_path in paths:
        print(str(loop_count) + ': ' + image_path)
        loop_count += 1

    image_path = paths[2]  #paths[int(input())]
    image = np.array(plt.imread(image_path))

    gridshape = image[:, :, 1].shape
    gridlength = gridshape[0]*gridshape[1]

    print(np.array(image[:, :, 1]))

    psi_rand = np.array(image[:, :, 0]).reshape(gridlength)*0x08
    u_rand = ((np.array(image[:, :, 1]).reshape(gridlength)*0xFF) - 0x7F) / 0x9F
    v_rand = ((np.array(image[:, :, 2]).reshape(gridlength)*0xFF) - 0x7F) / 0x9F
    b_rand = np.array(image[:, :, 3], dtype=bool)

    print(v_rand)

    dir_rand = getdirection(b_rand)

    xx = np.linspace(0, 1, gridshape[0])
    yy = np.linspace(0, 1, gridshape[1])
    XX, YY = np.meshgrid(xx, yy)

    WW = np.sin(4 * np.pi * (XX)) * np.sin(3 * np.pi * YY)
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
    Lap_rand = (h**(-2)) * (sparse.diags(dir_rand == 0x1, dtype=bool)*b1[0]  # S-Rand
                + sparse.diags(dir_rand == 0x2, dtype=bool)*b1[2]  # N-Rand
                + sparse.diags(dir_rand == 0x4, dtype=bool)*b1[1]  # O-Rand
                + sparse.diags(dir_rand == 0x8, dtype=bool) * b1[3]  # W-Rand  (heh SNOW...)

                + sparse.diags(dir_rand == 0x5, dtype=bool)*b2[0]  # SO
                + sparse.diags(dir_rand == 0x6, dtype=bool)*b2[1]  # NO
                + sparse.diags(dir_rand == 0x9, dtype=bool)*b2[3]  # SW
                + sparse.diags(dir_rand == 0xA, dtype=bool)*b2[2]  # SO
                )

    Neumann_Korrekturx = (1/h) * (sparse.diags(dir_rand & 0x1, dtype=float)
                              - sparse.diags(dir_rand & 0x2, dtype=float))

    Neumann_Korrektury = (1/h) * (sparse.diags(dir_rand & 0x4, dtype=float)
                              - sparse.diags(dir_rand & 0x8, dtype=float))

    del b1
    del b2

    Lap_rand = sparse.csr_matrix(Lap_rand)

    for i in [D1x,D1y,D1w,D1o,D1s,D1n,Lap0,Lap1]:
        i.final()
    Lap1i = splinalg.inv(Lap1.matrix)

options = wdg.RadioButtons(opt_ax, ('Euler-Vorwärtsschritt', 'RK4'))

text_box = wdg.TextBox(txt_ax, 'Anfangsbedingung', initial='numpy functions can be written as np.*')
text_box.on_submit(submit)

plt.show(fig)
