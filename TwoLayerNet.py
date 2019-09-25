import sys,os
import numpy as np
from collections import OrderedDict
from addplayer import Affine, relu, SoftmaxWithLoss
from numerical import numerical_diff
from dataset.mnist import load_mnist
from numerical_gradient import numerical_gradient


class TwoLayerNet:
    def __init__(self,input_size,hidden_size,output_size,weight_init_std=0.01):
        self.params={}
        self.params['W1']=weight_init_std*np.random.randn(input_size,hidden_size)
        self.params['b1']=weight_init_std*np.zeros(hidden_size)
        self.params['W2']=weight_init_std*np.random.randn(hidden_size,output_size)
        self.params['b2']=weight_init_std*np.zeros(output_size)
        #layer
        self.layers=OrderedDict()
        self.layers['Affine1']=Affine(self.params['W1'],self.params['b1'])
        self.layers['relu1']=relu()
        self.layers['Affine2']=Affine(self.params['W2'],self.params['b2'])
        self.lastlayer=SoftmaxWithLoss()
        
    def predict(self,x):
        for layer in self.layers.values():
            x=layer.forward(x)
        return x
    
    def loss(self,x,t):
        y=self.predict(x)
        return self.lastlayer.forward(y, t)
    
    def accuracy(self,x,t):
        y=self.predict(x)
        y=np.argmax(y,axis=1)
        if t.ndim != 1:t=np.argmax(t,axis=1)
        accuracy=np.sum(y==t)/float(x.shape[0])
        return accuracy
    
    def numerical_gradient(self,x,t):
        loss_W=lambda W:self.loss(x, t)
        grads={}
        grads['W1']=numerical_gradient(loss_W,self.params['W1'])
        grads['b1']=numerical_gradient(loss_W,self.params['b1'])
        grads['W2']=numerical_gradient(loss_W,self.params['W2'])
        grads['b2']=numerical_gradient(loss_W,self.params['b2'])
        return grads
    def gradient(self,x,t):
        self.loss(x, t)
        dout=1
        dout=self.lastlayer.backward(dout)
        layers=list(self.layers.values())
        layers.reverse()
        for layer in layers:
            #print("layer name:"+layer.)
            #print(dout.shape)
            dout=layer.backward(dout)
        
        grads={}
        grads['W1']=self.layers['Affine1'].dW
        grads['b1']=self.layers['Affine1'].db
        grads['W2']=self.layers['Affine2'].dW
        grads['b2']=self.layers['Affine2'].db
        return grads
'''
(x_train,t_train),(x_test,t_test)=load_mnist()
network=TwoLayerNet(input_size=784,hidden_size=100,output_size=10)
x_batch=x_train[:3]
t_batch=t_train[:3]
grad_numerical=network.numerical_gradient(x_batch, t_batch)
grad_backprop=network.gradient(x_batch,t_batch)
for key in grad_numerical.keys():
    diff=np.average(np.abs(grad_backprop[key]-grad_numerical[key]))
    print(key+":"+str(diff))
'''
(x_train,t_train),(x_test,t_test)=load_mnist()
network=TwoLayerNet(input_size=784,hidden_size=20,output_size=10) 
iters_num=100000
train_size=x_train.shape[0]
print(train_size)
batch_size=100
learning_rate=0.1
train_loss_list=[]
train_acc_list=[]
test_acc_list=[]
iter_per_epoch=max(train_size/batch_size, 1)
for i in range(iters_num):
    #print(i)
    batch_mask=np.random.choice(train_size,batch_size)
    x_batch=x_train[batch_mask]
    t_batch=t_train[batch_mask]
    grad=network.gradient(x_batch, t_batch)
    for key in ('W1','b1','W2','b2'):
        network.params[key]-=learning_rate*grad[key]
    loss=network.loss(x_batch,t_batch)
    train_loss_list.append(loss)
    if i % iter_per_epoch ==0:
        train_acc=network.accuracy(x_batch, t_batch)
        test_acc=network.accuracy(x_test, t_test)
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)
        print(train_acc,test_acc)
        print(network.params['b2'])
