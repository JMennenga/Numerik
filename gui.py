import matplotlib.pyplot as plt
import matplotlib.widgets as wdg
import numpy as np
import threading
import test

def submit(text):

    global drawobj
    global simloop_active

    lock = threading.Lock()

    draw_thread = threading.Thread(target = updateplot, args = (im,))
    sim_thread = threading.Thread(target = test.sim)

    draw_thread.start()
    sim_thread.start()

    draw_thread.join()
    sim_thread.join()
def updateplot(im):

    global drawobj
    global simloop_active

    while simloop_active:

        lock.acquire()
        w = drawobj
        lock.release()
        W = w.reshape(gridshape)

        W_oR = W[1:gridshape[0]-1,1:gridshape[1]-1]
        norm = colors.Normalize(np.min(WW_oR), np.max(WW_oR))

        im.set_data(W)
        im.set_norm(norm)
        plt.pause(0.0001)





fig = plt.figure(figsize=(12,9))

plt_frame = [0.15 ,0.15,0.8,0.8]
opt_frame =  [0.05,0.25, 0.1,0.2]
txt_frame =  [0.2, 0.05,0.7,0.05]

plt_ax = fig.add_subplot(111, position = plt_frame)
opt_ax = fig.add_axes(opt_frame, frameon = False)
txt_ax = fig.add_axes(txt_frame)

options = wdg.RadioButtons(opt_ax, ('Euler-Vorwärtsschritt', 'RK4'))

text_box = wdg.TextBox(txt_ax, 'Anfangsbedingung', initial='numpy functions can be written as np.*')
text_box.on_submit(submit)

im = plt.imshow
plt.show(fig)
