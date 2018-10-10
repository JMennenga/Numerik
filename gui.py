import os
import glob
import matplotlib.pyplot as plt
import matplotlib.widgets as wdg
import numpy as np
import threading
import time
import scipy.sparse as sparse
import scipy.sparse.linalg as splinalg
import matplotlib.colors as colors
from stencil import Ableitung

class plotwindow:
    __init__(self):
        self.frame = [0.15 ,0.15,0.8,0.8]
        self.ax = fig.add_subplot(111, position = plt_frame)
        self.Thread = threading.Thread(target = self.draw)
        self.active = False
        self.idle = threading.Event()

    draw(self):
        while self.active:
            if self.idle.is_set():
                self.Thread.wait()



def submit(text):
    def updateplot(im):

        global drawobj
        global simloop_active
        global gridshape

        barrier.wait()
        time.sleep(0.1)
        while not draw_terminate.is_set():

            lock.acquire()
            w = drawobj
            lock.release()
            W = w.reshape(gridshape)

            W_oR = W[1:gridshape[0]-1,1:gridshape[1]-1]
            norm = colors.Normalize(np.min(W_oR), np.max(W_oR))

            im.set_data(W)
            im.set_norm(norm)
            plt.pause(0.0001)

        draw_terminate.clear()


    try:
        sim_terminate.set()

    except NameError:
        print("No threads exists")
        sim_terminate = threading.Event()
        draw_terminate = threading.Event()


    lock = threading.Lock()
    barrier = threading.Barrier(2)
    draw_thread = threading.Thread(target = updateplot, args = (im,))
    sim_thread = threading.Thread(target = sim, args = (sim_terminate,))

    draw_thread.start()
    sim_thread.start()



global draw_thread
global drawobj


fig = plt.figure(figsize=(12,9))


opt_frame =  [0.05,0.25, 0.1,0.2]
txt_frame =  [0.2, 0.05,0.7,0.05]


opt_ax = fig.add_axes(opt_frame, frameon = False)
txt_ax = fig.add_axes(txt_frame)

options = wdg.RadioButtons(opt_ax, ('Euler-Vorwärtsschritt', 'RK4'))

text_box = wdg.TextBox(txt_ax, 'Anfangsbedingung', initial='numpy functions can be written as np.*')
text_box.on_submit(submit)
image = [[1,1],[1,0]]
im = plt_ax.imshow(image)
plt.show(fig)
