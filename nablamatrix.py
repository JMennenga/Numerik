import numpy as np
import scipy.linalg as lin

#Diskretisierung des Nabla-Operators bezueglich des Einheitsquadrats
def nablamatrixEQ(h):

    x=np.arange(h,1,h)
    X,Y=np.meshgrid(x,x)
    # F=np.vectorize(func)(X,Y).reshape(:,1)
    #F=np.array([func(x,y) for (x,y) in (X,Y)])
    #zu Testzwecken evtl hier returnen
    #return F
    schema=np.zeros(x.size[0],x.size[0])
    schema[0,0:2]=[-3,4,-1]
    schema[x.size[0]-1,x.size[0]-3:x.size[0]-1]=[1,-4,3]
    for i in range(1,x.size[0]-1):
        schema[i,i-1:i+1]=[-1,0,1]
    dx=(1/2*h)*np.kron(schema,np.eye(schema.size))
    dy=(1/2*h)*np.kron(np.eye(schema.size),schema)
    DX=np.matmul(dx,F)
    DY=np.matmul(dy,F)
    return DX,DY
def testfunc(arg1,arg2):
    return arg1 + arg2

print(nablaEQ(testfunc,0.1))
