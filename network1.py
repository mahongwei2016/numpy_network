import numpy as np
import matplotlib.pylab as plt
from sigmoid import sigmoid
x=np.array([1.0,0.5])
w1=np.array([[0.1,0.3,0.5],[0.2,0.4,0.6]])
b1=np.array([0.1,0.2,0.3])
a1=np.dot(x,w1)+b1
print(a1)
print(w1.shape)
print(x.shape)
print(b1.shape)
print(sigmoid(a1))