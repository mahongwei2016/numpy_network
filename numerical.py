import numpy as np
import matplotlib.pylab as plt
def function_1(x):
    return x**3
def function_2(x,k,b):
    return k*x+b
def function_3(f1,k,x):
    return f1(x)-k*x
def numerical_diff(f,x):
    h=0.0001
    return (f(x+h)-f(x-h))/(2*h)

def k_line(f1,x,c):
    k=numerical_diff(f1, c)
    b=function_3(f1,k,c)
    z=function_2(x,k,b)
    plt.plot(x,z,linestyle = "--",)


#x=np.arange(-5,5,0.1)
#y=function_1(x)
#plt.plot(x,y)
#k_line(function_1,x,0)
#plt.show()