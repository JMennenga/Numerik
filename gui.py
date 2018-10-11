import threading
import time
import wirbelstroemung

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
        image = [[1,1],[1,0]]
        self.im = self.ax.imshow(image)
        self.qu = self.ax.quiver()
        self.drawobj = []
        self.options = {
            'x'     : [0,1],
            'y'     : [0,1],
            'phi'   : False,
            'uv'    : False,
            'w'     : True
        }


    def draw(self):
        while self.active:
            if self.idle.is_set():
                self.Thread.wait()
            else:
                if options['w']:
                    self.im.set_da

    def start(self):
        self.active = True
        self.Thread.start()

    def setdrawobj(obj):
        self.lock.acquire()
        self.drawobj = object
        self.lock.release()

    def getdrawobj():
        self.lock.acquire()
        a = self.drawobj
        self.lock.release
        return a

def timestepselect(selected):
    global optionselected
    optiondict = {'Euler-Vorwärtsschritt' : 0, 'rk4' : 1}
    timestepselected = optiondict[selected]

def orderselect(selected):
    global orderselected
    orderdict = {'1st Order' : 0, 'Higher Order' : ord}
    orderselected = orderdict[selected]

def randload(callevent):

    global simobj

    tmp = Tk()
    tmp.withdraw() # we don't want a full GUI, so keep the root window from appearing
    filename = filedialog.askopenfilename() # show an "Open" dialog box and return the path to the selected file
    tmp.destroy()

    simobj = wirbelstroemung.Wirbelstroemung(filename)

def text_box_changed(text):
    global simobj
    simobj.w0string = text

def simstart(callevent):
    def sim(wirbel_obj, sim_terminate):
        global plot

        wirbel_obj.setup()
        for a in wirbel_obj.rk4(sim_terminate, wirbel_obj.rhs, wirbel_obj.get_w0()):
            plot.setdrawobj(a)


    global simobj
    global sim_thread
    global sim_terminate

    try:
        sim_terminate.set()
    except NameError:
        sim_terminate = threading.Event()

    sim_thread = threading.Thread(target = sim, args = (simobj, sim_terminate))
    sim_thread.start()

    simobj.setup()


global simobj
global sim_thread
global sim_terminate
global sim_options

global plot
global plot_options

sim_options = {
'path' : '',
'timestep' : 0,
'order' : 0,
'text' : 'np.sin(3 * XX) * np(2 * YY)',
'h': 0.1,
'kin_vis' : 0.1,
'CFL' : 0.9
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

plt.show(fig)
