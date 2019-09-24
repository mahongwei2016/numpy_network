import sys,os
import numpy as np
from softmax import softmax
from lose import cross_entropy_error
from numerical_gradient import numerical_gradient



class simpleNet:
    def __init__(self):
        self.W=np.random.randn(2,3)
    
    def predict(self,x):
        return np.dot(x,self.W)
    
    def loss(self,x,t):
        z=self.predict(x)
        y=softmax(z)
        loss=cross_entropy_error(y,t)
        return loss
    
    

net=simpleNet()
print(net.W)
x=np.array([0.6,0.9])
p=net.predict(x)
print(p)
print(np.argmax(p))
t=np.array([0,0,1])
print(net.loss(x, t))
def f(W):
    return net.loss(x, t)
print(net.W.shape)
dw=numerical_gradient(f,net.W)
print(dw)