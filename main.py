import numpy as np
import threading
import wirbelstroemung
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from tkinter import Tk, filedialog

global sim_stop
global sim_started
global draw_lock
global drawobj
global w0
global b

sim_options = {
'image' : [],
'timestep' : 0,
'order' : 10,
'text' : 'np.sin(3 * np.pi * XX) * np.sin(4 * np.pi * YY)',
'h':   1,
'kin_vis' : 0.1,
'CFL' : 0.7,
'inverted' : True
}


tmp = Tk()
tmp.withdraw() # we don't want a full GUI, so keep the root window from appearing
sim_options['image'] = np.array(plt.imread(filedialog.askopenfilename())) # show an "Open" dialog box and return the path to the selected file
tmp.destroy()
sim_options['shape'] = sim_options['image'].shape[:2]
#Thread setup
draw_lock = threading.Lock()
sim_stop = threading.Event()
sim_started = threading.Event()
b = threading.Barrier(2)
simobj = wirbelstroemung.Wirbelstroemung(sim_options)
simobj.setup()
w0 = simobj.get_w0()
def sim(wirbel_obj):

    global sim_stop
    global sim_started
    global draw_lock
    global drawobj
    global b
    global w0

    for w in wirbel_obj.rk4(sim_stop, wirbel_obj.rhs, w0):
        draw_lock.acquire()
        drawobj = w
        draw_lock.release()
        sim_started.set()
        b.wait()
    sim_started.clear()

sim_Thread = threading.Thread(target = sim, args=(simobj,))
sim_Thread.start()

#Window setup
plt.ion()
window = plt.figure(figsize=(10,8))
plot = window.add_subplot(111)
im = plot.imshow(w0.reshape(sim_options['shape']))
cbar = plt.colorbar(im)

def window_closed(evt):
    global sim_stop
    sim_stop.set()

window.canvas.mpl_connect('close_event', window_closed)
plt.show(block = False)
plt.pause(0.001)

sim_started.wait()
while sim_started.is_set():
    b.wait()
    draw_lock.acquire()
    ww = drawobj[1]
    draw_lock.release()
    ww = ww.reshape(sim_options['shape'])
    im.set_data(ww)
    norm = colors.Normalize(np.min(ww[1:sim_options['shape'][0]-1,1:sim_options['shape'][1]-1]),
                            np.max(ww[1:sim_options['shape'][0]-1,1:sim_options['shape'][1]-1]))
    im.set_norm(norm)
    plt.pause(0.001)
sim_Thread.join()
