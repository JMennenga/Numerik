import os
import shutil
import copy
import numpy as np
import threading
import wirbelstroemung
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.animation as ani
import matplotlib.gridspec as gridspec
from tkinter import Tk, filedialog
plt.rcParams['animation.ffmpeg_path'] = 'C:/Users/Marc/workspace/FFmpeg/bin/ffmpeg'

global sim_stop
global sim_started
global draw_lock
global drawobj
global w0
global b

sim_options = {
'image' : [],
'timestep' : 0,
'order' : 9,
'text' : '5 * np.sin(3 * np.pi * XX)**2 * np.sin(3 * np.pi * YY) * np.exp(-(XX-0.6)**2 - (YY-0.7)**2)' ,
'l':   1,
'kin_vis' : 0.0005,  #stabilitätsprobleme bei O(h)^2 ~ O(kin_vis)
'CFL' : 0.8,
'inverted' : True
}
#Instabilitöt liegt vermutlich an der Ausführung des Zeitschritts

tmp = Tk()
tmp.withdraw() # we don't want a full GUI, so keep the root window from appearing
sim_options['image'] = np.array(plt.imread(filedialog.askopenfilename())) # show an "Open" dialog box and return the path to the selected file

tmp.destroy()
sim_options['shape'] = sim_options['image'].shape[:2]
sim_options['h'] = sim_options['l'] / (sim_options['shape'][0]-1)

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
        #b.wait()
    sim_started.clear()
    b.abort()

sim_Thread = threading.Thread(target = sim, args=(simobj,))
sim_Thread.start()

#Window setup
window = plt.figure(figsize=(10,9))
gc = gridspec.GridSpec(4, 1)
plot = window.add_subplot(gc[0:3,0])
plot.set_title('Wirbelstärke')
im = plot.imshow(w0.reshape(sim_options['shape']))
# norm = colors.Normalize(vmin = np.min(w0),
#                             vmax = np.max(w0))
norm = colors.SymLogNorm(0.01,  vmin = np.min(w0),
                            vmax = np.max(w0))
im.set_norm(norm)
cbar = plt.colorbar(im)

global histo_data

histo_plot = window.add_subplot(gc[3,0])
histo_data = np.array([[0,0]])#[[0,w0.sum()*sim_options['h']**2]])
histo_line, = histo_plot.plot(histo_data[:,0],histo_data[:,1])

def window_closed(evt):
    global sim_stop
    global sim_Thread
    sim_stop.set()
    b.abort()
    #b.wait()
    print('STOP')

window.canvas.mpl_connect('close_event', window_closed)

sim_started.wait()

def threadread():
    global b
    global draw_lock
    global drawobj
    try:
        while sim_started.is_set():
            #b.wait()
            draw_lock.acquire()
            c = copy.deepcopy(drawobj)
            draw_lock.release()
            yield c
    except threading.BrokenBarrierError:
        yield c
def animation(frame):
    global histo_data
    global simobj
    t = frame[0]
    ww = frame[1]
    histo_data = np.append(histo_data,[[t,ww.sum()*sim_options['h']**2]], 0)
        #ww[np.logical_not(simobj.b_rand)].sum()*sim_options['h']**2]], 0)

    histo_line.set_ydata(histo_data[:,1])
    histo_line.set_xdata(histo_data[:,0])
    histo_plot.set_xlim([0,histo_data[-1,0]])
    histo_plot.set_ylim(np.min([0] + histo_data[:,1]) ,max([0] + histo_data[:,1]))

    ww_n = ww[np.logical_not(simobj.b_rand)]
    ww = ww.reshape(sim_options['shape'])
    im.set_data(ww)
    # norm = colors.Normalize(vmin = np.min(ww_n),
    #                         vmax = np.max(ww_n))
    norm = colors.SymLogNorm(0.01,  vmin = np.min(ww_n),
                           vmax = np.max(ww_n))
    im.set_norm(norm)

    if t >= 30:
        a = open('../30s Werte.csv', 'a')
        a.write(str(sim_options['order']))
        a.write(str(sim_options['h']))
        a.write(str(histo_data[1]))
        a.write(str(histo_data[-1]))
        sim_stop.set()

    return [histo_line,im]

anim = ani.FuncAnimation(window, animation, frames=threadread)
FFwriter = ani.FFMpegWriter(fps = 30)

i = 0

os.mkdir('tmp')
os.chdir('tmp')

while not sim_stop.is_set():
    anim.save('cc19_50_' + str(i) + '.mp4', writer = FFwriter)
    i += 1

os.system("(for %i in (*.mp4) do @echo file '%i') > mylist.txt")
os.system("ffmpeg -f concat -i mylist.txt -c copy vid.mp4")
os.system("move vid.mp4 ..")
os.chdir("..")
shutil.rmtree('tmp')
plt.show()
sim_Thread.join()
