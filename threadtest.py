import threading

def Threadingtest():
    def a(a_terminate, b_terminate):

        while not a_terminate.is_set():
            for a in range(10):
                c = a
        b_terminate.set()
        a_terminate.clear()

    def b(b_terminate):
        while not b_terminate.is_set():
            for a in range(10):
                c = a
        b_terminate.clear()

    global a_terminate
    global b_terminate
    global A
    global B

    try:
        a_terminate.set()
        B.join()
    except NameError:
        print('nich ok')

        a_terminate = threading.Event()
        b_terminate = threading.Event()
    else:
        print('ok')
    A = threading.Thread(target = a, name = 'a', args=(a_terminate,b_terminate))
    B = threading.Thread(target = b, name = 'b', args=(b_terminate,))


    A.start()
    B.start()

    print(threading.enumerate())


Threadingtest()

Threadingtest()
