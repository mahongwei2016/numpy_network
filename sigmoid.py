import numpy as np
import matplotlib.pylab as plt
def sigmoid(x):
    return 1/(1+np.exp(-x))
def sigmoid_grad(x):
    return (1.0-sigmoid(x))*sigmoid(x)
#x=np.arange(-5.0,5.0,0.1)
#y=sigmoid(x)
#plt.plot(x,y)
#plt.ylim(-0.1,1.1)
#plt.show()
