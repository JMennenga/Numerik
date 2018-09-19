import numpy as np

def vorrueck(L,R,b):
    if np.size(L)==np.size(R) and np.size(b)==(np.size(L)[0],1) and np.size(L)[0]==np.size(L)[1]:
        y=np.zeros(np.size(b))
<<<<<<< HEAD
=======
        x=np.zeros(np.size(b))
>>>>>>> 7bf927d9fe63dc782002f4ee800d92c506d9fb33
#vorwaertseinsetzen
        for i in np.arange(np.size(L)[0]):
            for j in np.arange(i):
                y[i]-=L[j,i]*y[j]
            y[i]+=b[i]
            y[i]=y[i]/L[i,i]
#rueckwaertseinsetzen
        for i in np.arange(np.size(L)[0]-1,-1,-1):
            for j in np.arange(i-1,-1,-1):
<<<<<<< HEAD
                x[i]-=L[j,i]*x[j]
            x[i]+=y[i]
            x[i]=x[i]/L[i,i]
=======
                y[i]-=L[j,i]*y[j]
            x[i]+=y[i]
            y[i]=y[i]/L[i,i]
        return x
    else: print("Ein/e Vektor/Matrix hat die falsche Dimension")
>>>>>>> 7bf927d9fe63dc782002f4ee800d92c506d9fb33
