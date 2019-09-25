import numpy as np
from mulayer import Mullayer
import mulayer
from softmax import softmax
from lose import cross_entropy_error

class AddLayer:
    def __init__(self):
        pass
    def forward(self,x,y):
        out=x+y
        return out
    
    def backout(self,dout):
        dx=dout*1
        dy=dout*1
        return dx,dy


class relu:
    def __init__(self):
        self.mask=None
    def forward(self,x):
        self.mask=(x<=0)
        out=x.copy()
        out[self.mask]=0
        return out
    
    def backward(self,dout):
        #print(dout.shape)
        dout[self.mask]=0
        dx=dout
        return dx

class sigmoid:
    def __init__(self):
        self.out=None
    def forward(self,x):
        out=1/(1+np.exp(-x))
        self.out=out
        return out
    def backward(self,dout):
        dx=dout*(1.0-self.out)*self.out
        return dx

class Affine:
    def __init__(self, W, b):
        self.W =W
        self.b = b
        
        self.x = None
        self.original_x_shape = None
        # 权重和偏置参数的导数
        self.dW = None
        self.db = None

    def forward(self, x):
        # 对应张量
        self.original_x_shape = x.shape
        x = x.reshape(x.shape[0], -1)
        self.x = x

        out = np.dot(self.x, self.W) + self.b

        return out

    def backward(self, dout):
        dx = np.dot(dout, self.W.T)
        self.dW = np.dot(self.x.T, dout)
        self.db = np.sum(dout, axis=0)
        
        dx = dx.reshape(*self.original_x_shape)  # 还原输入数据的形状（对应张量）
        return dx
        
class Affine1:
    def __init__(self,W,b):
        self.W=W
        self.b=b
        self.x=None
        self.dW=None
        self.db=None
        
    def forward(self,x):
        self.x=x
        out=np.dot(x,self.W)+self.b
        return out
    def backward(self,dout):
        dx=np.dot(dout,self.W.T)
        self.dW=np.dot(self.x.T,dout)
        self.db=np.sum(dout,axis=0)
        
class SoftmaxWithLoss:
    def __init__(self):
        self.loss=None
        self.y=None
        self.t=None
    def forward(self,x,t):
        self.t=t
        self.y=softmax(x)
        self.loss=cross_entropy_error(self.y,self.t)
        return self.loss
    def backward1(self,dout=1):
        batch_size=self.t.shape[0]
        dx=(self.y-self.t)/batch_size
        return dx  
    def backward(self, dout=1):
        batch_size = self.t.shape[0]
        if self.t.size == self.y.size: # 监督数据是one-hot-vector的情况
            dx = (self.y - self.t) / batch_size
        else:
            dx = self.y.copy()
            dx[np.arange(batch_size), self.t] -= 1
            dx = dx / batch_size
        
        return dx
apple=100
apple_num=2
orange=150
orange_num=3
tax=1.1
#layer
mul_apple_layer=Mullayer()
mul_orange_layer=Mullayer()
add_apple_orange_layer=AddLayer()
mul_tax_layer=Mullayer()

#forward
apple_price=mul_apple_layer.forward(apple,apple_num)
orange_price=mul_orange_layer.forward(orange, orange_num)
all_price=add_apple_orange_layer.forward(apple_price, orange_price)
price=mul_tax_layer.forward(all_price, tax)

#backward
dprice=1
dall_price,dtax=mul_tax_layer.backward(dprice)
dapple_price,dorange_price=add_apple_orange_layer.backout(dall_price)
dorange,dorange_num=mul_orange_layer.backward(dorange_price)
dapple,dapple_num=mul_apple_layer.backward(dapple_price)
'''
print(apple_price,orange_price,all_price,price)
print(dall_price,dtax,dapple_price,dorange_price,dorange,dorange_num,dapple,dapple_num)

x=np.array([[1.0,-0.5],[-2.0,3.0]])
print(x)
mask=(x<0)
print(mask)

relu_layer=relu()
yrelu=relu_layer.forward(x)
print(yrelu)
dx=relu_layer.backward(yrelu)
print(dx)

softmax_layer=sigmoid()
ysoftmax=softmax_layer.forward(x)
print(ysoftmax)
dx=softmax_layer.backward(ysoftmax)
print(dx)


X=np.random.rand(2)
W=np.random.rand(2,3)
B=np.random.rand(3)
Y=np.dot(X,W)+B
print(Y)
'''