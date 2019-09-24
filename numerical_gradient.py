import numpy as np
import matplotlib.pylab as plt
from numerical import numerical_diff
def function_4(x):
    return x[0]**2+x[1]**2

def function_tmp1(x0):
    return x0*x0+4.0**2.0

def function_tmp2(x1):
    return 3.0**2.0+x1**2

def numerical_gradient(f,x):
    h=1e-4
    grad=np.zeros_like(x)
    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        idx = it.multi_index
        tmp_val=x[idx]
        x[idx]=tmp_val+h
        fxh1=f(x)
        x[idx]=tmp_val-h
        fxh2=f(x)
        grad[idx]=(fxh1-fxh2)/(2*h)
        x[idx]=tmp_val
        it.iternext() 
    return grad
        
def gradient_descent(f,init_x,lr=0.01,step_num=100):
    x=init_x
    for i in range(step_num):
        grad=numerical_gradient(f, x)
        x-=lr*grad
    return x

#print(numerical_diff(function_tmp1,3.0))
#print(numerical_diff(function_tmp2,4.0))


#print(numerical_gradient(function_4,np.array([3.0,4.0])))

#init_x=np.array([-3.0,4.0])
#print(gradient_descent(function_4,init_x=init_x,lr=0.1,step_num=100))