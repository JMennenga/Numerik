import numpy as np

def vorrueck(L,R,b):
    if np.size(L)==np.size(R) and np.size(b)=(np.size(L)[0],1) and np.size(L)[0]==np.size(L)[1]:
        y=np.zeros(np.size(b))
#vorwaertseinsetzen
        for i in np.arange(np.size(L)[0]):
            for j in np.arange(i):
                y[i]-=L[j,i]*y[j]
            y[i]+=b[i]
            y[i]=y[i]/L[i,i]
#rueckwaertseinsetzen
        for i in np.arange(np.size(L)[0]-1,-1,-1):
            for j in np.arange(i-1,-1,-1):
                y[i]-=L[j,i]*y[j]
            x[i]+=y[i]
            y[i]=y[i]/L[i,i]
