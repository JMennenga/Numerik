import threading
import time
import wirbelstroemung

import numpy as np

import matplotlib.colors as colors
import matplotlib.pyplot as plt
import matplotlib.widgets as wdg

from stencil import Ableitung
from tkinter import Tk, filedialog

class plotwindow:
    def __init__(self):
        self.frame = [0.15, 0.15, 0.8, 0.8]
        self.ax = fig.add_subplot(111, position = self.frame)
        self.Thread = threading.Thread(target = self.draw)
        self.lock = threading.Lock()
        self.active = False
        self.idle = threading.Event()
        image = [[1,1],
                [1,0]]

        self.ax.set_xlim((0,5))
        self.ax.set_ylim((5,0))
        self.im = self.ax.imshow(image, extent = [0,5,5,0])
        self.qu = self.ax.quiver(image,image)
        self.co = self.ax.contour(image)
        self.drawobj = [-1, image, image, image, image]
        self.options = {
            'size'  : (1, 1),
            'x'     : [0, 1],
            'y'     : [0, 1],
            'phi'   : False,
            'uv'    : False,
            'w'     : True
        }


    def draw(self):
        while self.active:
            print(self.idle.is_set())
            self.idle.wait()
            drawobj = self.getdrawobj()
            if drawobj[0] != -1:
                print('a')
                self.ax.clear()
                if self.options['w']:
                    self.ax.imshow(drawobj[4], extent = [0,self.options['size'][0],0,self.options['size'][1]])

                if self.options['uv']:
                    self.ax.quiver(self.options['x'], self.options['y'], drawobj[2],drawobj[3])

                if self.options['phi']:
                    self.ax.contour(drawobj[1])

            # plt.pause(0.001)

    def start(self):
        self.active = True
        self.Thread.start()

    def setdrawobj(self, obj):
        self.lock.acquire()
        self.drawobj = object
        self.lock.release()

    def getdrawobj(self):
        self.lock.acquire()
        a = self.drawobj
        self.lock.release
        return a


def timestepselect(selected):
    optiondict = {'Euler-Vorwärtsschritt' : 0, 'rk4' : 1}
    sim_options['timestep'] = optiondict[selected]

def orderselect(selected):
    global orderselected
    orderdict = {'1st Order' : 0, 'Higher Order' : ord}
    orderselected = orderdict[selected]

def randload(callevent):
    global sim_options
    global plot

    tmp = Tk()
    tmp.withdraw() # we don't want a full GUI, so keep the root window from appearing
    sim_options['image'] = np.array(plt.imread(filedialog.askopenfilename())) # show an "Open" dialog box and return the path to the selected file
    tmp.destroy()

def text_box_changed(text):
    global simobj
    simobj.w0string = text

def simstart(callevent):

    def sim(sim_terminate):

        global plot
        global simobj
        sim_terminate.clear()
        simobj.setup()
        plot.idle.set()
        for a in simobj.rk4(sim_terminate, simobj.rhs, simobj.get_w0()):
            plot.setdrawobj(a)
        plot.idle.clear()


    global simobj
    global sim_thread
    global sim_terminate
    global sim_options

    global plot

    try:
        sim_terminate.set()
    except NameError:
        sim_terminate = threading.Event()

    simobj = wirbelstroemung.Wirbelstroemung(sim_options)
    sim_thread = threading.Thread(target = sim, args = (sim_terminate,))
    sim_thread.start()

    plot.options['size'] = (sim_options['h'] * sim_options['image'].shape[0],
                            sim_options['h'] * sim_options['image'].shape[1])


global simobj
global sim_thread
global sim_terminate
global sim_options

global plot
global plot_options

sim_options = {
'image' : [],
'timestep' : 0,
'order' : 10,
'text' : 'np.sin(3 * XX) * np.sin(2 * YY)',
'h': 0.1,
'kin_vis' : 0.1,
'CFL' : 0.9,
'inverted' : 0
}


fig = plt.figure(figsize=(12,9))

opttime_frame =  [0.05,0.25, 0.15, 0.15]
optorder_frame = [0.05, 0.5, 0.15, 0.15]

txt_frame =  [0.2, 0.05,0.65,0.05]
loadbtn_frame = [0.05,0.9,0.1,0.05]
gobtn_frame = [0.9, 0.05, 0.05, 0.05]


opttime_ax = fig.add_axes(opttime_frame, frameon = False)
optorder_ax = fig.add_axes(optorder_frame, frameon = False)


txt_ax = fig.add_axes(txt_frame)
loadbtn_ax = fig.add_axes(loadbtn_frame)
gobtn_ax = fig.add_axes(gobtn_frame)

options_timestep = wdg.RadioButtons(opttime_ax, ('Euler-Vorwärtsschritt', 'RK4'))
options_timestep.on_clicked(timestepselect)

options_order = wdg.RadioButtons(optorder_ax, ('1st Order', 'Higher Order'))
options_order.on_clicked(orderselect)

text_box = wdg.TextBox(txt_ax, 'Anfangsbedingung', initial=sim_options['text'])
text_box.on_text_change(text_box_changed)

load_btn = wdg.Button(loadbtn_ax, 'Rand auswählen')
load_btn.on_clicked(randload)

go_btn = wdg.Button(gobtn_ax, 'Start')
go_btn.on_clicked(simstart)

plot = plotwindow()
plot.start()

plt.ion()
plt.show(fig)
