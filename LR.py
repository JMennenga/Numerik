import numpy as np


Test = np.matrix([[2,3,5],[6,12,22],[4,12,20]]


def LR(R):
    if np.size(R)[0]==np.size(R)[1]:
        L=np.eye(np.size(R))
        for i in range(np.size(R)[0]):
            for j in range(i+1,np.size(R)[1]):
                a=R[j.i]/R[i,i]
                L[j,i]=-a
                for k in range(i+1,np.size(R)[1]):
                    R[j,k]=R[j,k]-a*R[i,k]
        return (L,R)
    else:
        print("Matrix nicht quadratisch")
        return 0

(L,R)=LR(Test)
print("L")
print(L)
print("R")
print(R)
