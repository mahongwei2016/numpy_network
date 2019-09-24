import sys,os
from orca.scripts import self_voicing
from sigmoid import sigmoid
from softmax import softmax
from numerical_gradient import numerical_gradient
sys.path.append(os.pardir)
import numpy as np
class twolayernet:
    def __init__(self,input_size,hidden_size,output_size,weight_init_std=0.01):
        self.params ={}
        self.params['w1']=weight_init_std * np.random.randn(input_size,hidden_size)
        self.params['b1']=np.zeros(hidden_size)
        self.params['w2']=weight_init_std * np.random.randn(hidden_size,output_size)
        self.params['b2']=np.zeros(output_size)
        
    def predict(self,x):
        w1,w2=self.params['w1'],self.params['w2']
        b1,b2=self.params['b1'],self.params['b2']
        
        a1=np.dot(x,w1)+b1
        z1=sigmoid(a1)
        a2=np.dot(z1,w2)+b2
        y=softmax(a2)
        return y
    
    def loss(self,x,t):
        y=self.predict(x)
        y=np.argmax(y,axis=1)
        t=np.argmax(t,axis=1)
        accurary=np.sum(y==t)
        accuracy=np.sum(y==t)/float(x.shape[0])
        return accuracy
    
    def numerical_gradient(self,x,t):
        loss_w=lambda w: self.loss(x, t)
        grads={}
        grads['w1']=numerical_gradient(loss_w,self.params['w1'])
        grads['b1']=numerical_gradient(loss_w,self.params['b1'])
        grads['w2']=numerical_gradient(loss_w,self.params['w2'])
        grads['b2']=numerical_gradient(loss_w,self.params['b2'])
        return grads
    
    
#net=twolayernet(input_size=784,hidden_size=100,output_size=10)
#print(net.params['w1'].shape)
#print(net.params['b1'].shape)
#print(net.params['w2'].shape)
#print(net.params['b1'].shape)
#x=np.random.rand(100,784)
#y=net.predict(x)
#print(y.shape)


        