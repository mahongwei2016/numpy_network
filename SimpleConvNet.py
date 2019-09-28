import numpy as np
from collections import OrderedDict
from addplayer import relu, Affine, SoftmaxWithLoss
from common.util import im2col, col2im
from numerical_gradient import numerical_gradient
import pickle
import matplotlib.pyplot as plt
from dataset.mnist import load_mnist
from common.optimizer import *
class Convolution:
    def __init__(self,W,b,stride=1,pad=0):
        self.W=W
        self.b=b
        self.stride=stride
        self.pad=pad
        
        self.x=None
        self.col=None
        self.col_W=None
        self.dW=None
        self.db=None
        
        
    def forward(self,x):
        FN,C,FH,FW=self.W.shape
        N,C,H,W=x.shape 
        out_h=1+int((H+2*self.pad-FH)/self.stride)
        out_w=1+int((W+2*self.pad-FW)/self.stride)
        col=im2col(x,FH,FW,self.stride,self.pad)
        col_W=self.W.reshape(FN,-1).T
        out=np.dot(col,col_W)+self.b
        out=out.reshape(N,out_h,out_w,-1).transpose(0,3,1,2)
        self.x=x 
        self.col=col 
        self.col_W=col_W
        return out
    
    def backward(self,dout):
        FN,C,FH,FW=self.W.shape 
        dout=dout.transpose(0,2,3,1).reshape(-1,FN)
        self.db=np.sum(dout,axis=0)
        self.dW=np.dot(self.col.T,dout)
        self.dW=self.dW.transpose(1,0).reshape(FN,C,FH,FW)
        dcol = np.dot(dout,self.col_W.T)
        dx=col2im(dcol, self.x.shape, FH, FW , self.stride, self.pad)
        return dx 
    
class Pooling:
    def __init__(self,pool_h,pool_w,stride=1,pad=0):
        self.pool_h=pool_h
        self.pool_w=pool_w
        self.stride=stride
        self.pad=pad
        self.x=None
        self.arg_max=None
        
    def forward(self,x):
        N,C,H,W=x.shape
        out_h=int(1+(H-self.pool_h)/self.stride)
        out_w=int(1+(W-self.pool_w)/self.stride)
        col=im2col(x,self.pool_h,self.pool_w,self.stride,self.pad)
        col=col.reshape(-1,self.pool_h*self.pool_w)
        arg_max=np.argmax(col,axis=1)
        out=np.max(col,axis=1)
        out=out.reshape(N,out_h,out_w,C).transpose(0,3,1,2)
        self.x=x 
        self.arg_max=arg_max
        return out
    def backward(self,dout):
        dout=dout.transpose(0,2,3,1)
        pool_size=self.pool_h*self.pool_w
        dmax=np.zeros((dout.size,pool_size))
        dmax[np.arange(self.arg_max.size), self.arg_max.flatten()] = dout.flatten()
        dmax = dmax.reshape(dout.shape + (pool_size,)) 
        
        dcol = dmax.reshape(dmax.shape[0] * dmax.shape[1] * dmax.shape[2], -1)
        dx = col2im(dcol, self.x.shape, self.pool_h, self.pool_w, self.stride, self.pad)
        
        return dx

        
class SimpleConvNet:
    def __init__(self,input_dim=(1,28,28),\
                 conv_param={'filter_num':30,'filter_size':5,'pad':0,'stride':1},\
                 hidden_size=100,output_size=10,weight_init_std=0x01):
        filter_num=conv_param['filter_num']
        filter_size=conv_param['filter_size']
        filter_pad=conv_param['pad']
        filter_stride=conv_param['stride']
        input_size=input_dim[1]
        conv_output_size=(input_size-filter_size+2*filter_pad)/filter_stride +1
        pool_output_size=int(filter_num*(conv_output_size/2)*(conv_output_size/2))
        self.params={}
        self.params['W1']=weight_init_std*np.random.randn(filter_num,input_dim[0],filter_size,filter_size)
        self.params['b1']=np.zeros(filter_num)
        self.params['W2']=weight_init_std*np.random.randn(pool_output_size,hidden_size)
        self.params['b2']=np.zeros(hidden_size)
        self.params['W3']=weight_init_std*np.random.randn(hidden_size,output_size)
        self.params['b3']=np.zeros(output_size)
        self.layers=OrderedDict()
        self.layers['Conv1']=Convolution(self.params['W1'],\
                                         self.params['b1'],\
                                         conv_param['stride'],\
                                         conv_param['pad'] )
        self.layers['Relu1']=relu()
        self.layers['Pool1']=Pooling(pool_h=2,pool_w=2,stride=2)
        self.layers['Affine1']=Affine(self.params['W2'],self.params['b2'])
        self.layers['Relu2']=relu()
        self.layers['Affine2']=Affine(self.params['W3'],self.params['b3'])
        self.last_layer=SoftmaxWithLoss()
    def predict(self,x):
        for layer in self.layers.values():
            x=layer.forward(x)
        return x 
    def loss(self,x,t):
        y=self.predict(x)
        return self.last_layer.forward(y, t)
    
    def accuracy(self,x,t,batch_size=100):
        if t.ndim!=1:t=np.argmax(t,axis=1)
        acc=0.0
        for i in range(int(x.shape[0]/batch_size)):
            tx=x[i*batch_size:(i+1)*batch_size]
            tt=t[i*batch_size:(i+1)*batch_size]
            y=self.predict(tx)
            y=np.argmax(y,axis=1)
            acc+=np.sum(y==tt)
        return acc/x.shape[0]
    
    def numerical_gradient(self,x,t):
        loss_w=lambda w:self.loss(x,t)
        grads={}
        for idx in (1,2,3):
            grads['W'+str(idx)]=numerical_gradient(loss_w, self.params['W' + str(idx)])
            grads['b'+str(idx)]=numerical_gradient(loss_w, self.params['b' + str(idx)])
        
        return grads
    
    def gradient(self,x,t):
        #forward
        self.loss(x,t)
        
        #backward
        dout=1
        dout=self.last_layer.backward(dout)
        layers=list(self.layers.values())
        layers.reverse()
        for layer in layers:
            dout=layer.backward(dout)
        grads={}
        grads['W1']=self.layers['Conv1'].dW
        grads['b1']=self.layers['Conv1'].db
        grads['W2']=self.layers['Affine1'].dW
        grads['b2']=self.layers['Affine1'].db
        grads['W3']=self.layers['Affine2'].dW
        grads['b3']=self.layers['Affine2'].db
        return grads
    
    def save_params(self,file_name="params.pkl"):
        params={}
        for key,val in self.params.items():
            params[key]=val
        with open(file_name,'wb') as f:
            pickle.dump(params,f)
        
    def load_params(self,file_name="params.pkl"):
        with open(file_name,'rb') as f:
            params=pickle.load()
        for key,val in params.items():
            self.params[key]=val
        for i ,key in enumerate(['Conv1', 'Affine1', 'Affine2']):
            self.layers[key].W = self.params['W' + str(i+1)]
            self.layers[key].b = self.params['b' + str(i+1)]


class Trainer:
    """进行神经网络的训练的类
    """
    def __init__(self, network, x_train, t_train, x_test, t_test,
                 epochs=20, mini_batch_size=100,
                 optimizer='SGD', optimizer_param={'lr':0.01}, 
                 evaluate_sample_num_per_epoch=None, verbose=True):
        self.network = network
        self.verbose = verbose
        self.x_train = x_train
        self.t_train = t_train
        self.x_test = x_test
        self.t_test = t_test
        self.epochs = epochs
        self.batch_size = mini_batch_size
        self.evaluate_sample_num_per_epoch = evaluate_sample_num_per_epoch

        # optimzer
        optimizer_class_dict = {'sgd':SGD, 'momentum':Momentum, 'nesterov':Nesterov,
                                'adagrad':AdaGrad, 'rmsprpo':RMSprop, 'adam':Adam}
        self.optimizer = optimizer_class_dict[optimizer.lower()](**optimizer_param)
        
        self.train_size = x_train.shape[0]
        self.iter_per_epoch = max(self.train_size / mini_batch_size, 1)
        self.max_iter = int(epochs * self.iter_per_epoch)
        self.current_iter = 0
        self.current_epoch = 0
        
        self.train_loss_list = []
        self.train_acc_list = []
        self.test_acc_list = []

    def train_step(self):
        batch_mask = np.random.choice(self.train_size, self.batch_size)
        x_batch = self.x_train[batch_mask]
        t_batch = self.t_train[batch_mask]
        
        grads = self.network.gradient(x_batch, t_batch)
        self.optimizer.update(self.network.params, grads)
        
        loss = self.network.loss(x_batch, t_batch)
        self.train_loss_list.append(loss)
        if self.verbose: print("train loss:" + str(loss))
        
        if self.current_iter % self.iter_per_epoch == 0:
            self.current_epoch += 1
            
            x_train_sample, t_train_sample = self.x_train, self.t_train
            x_test_sample, t_test_sample = self.x_test, self.t_test
            if not self.evaluate_sample_num_per_epoch is None:
                t = self.evaluate_sample_num_per_epoch
                x_train_sample, t_train_sample = self.x_train[:t], self.t_train[:t]
                x_test_sample, t_test_sample = self.x_test[:t], self.t_test[:t]
                
            train_acc = self.network.accuracy(x_train_sample, t_train_sample)
            test_acc = self.network.accuracy(x_test_sample, t_test_sample)
            self.train_acc_list.append(train_acc)
            self.test_acc_list.append(test_acc)

            if self.verbose: print("=== epoch:" + str(self.current_epoch) + ", train acc:" + str(train_acc) + ", test acc:" + str(test_acc) + " ===")
        self.current_iter += 1

    def train(self):
        for i in range(self.max_iter):
            self.train_step()

        test_acc = self.network.accuracy(self.x_test, self.t_test)

        if self.verbose:
            print("=============== Final Test Accuracy ===============")
            print("test acc:" + str(test_acc))


(x_train, t_train), (x_test, t_test) = load_mnist(flatten=False)       
max_epochs = 20
network = SimpleConvNet(input_dim=(1,28,28), 
                        conv_param = {'filter_num': 30, 'filter_size': 5, 'pad': 0, 'stride': 1},
                        hidden_size=100, output_size=10, weight_init_std=0.01)   
            
trainer = Trainer(network, x_train, t_train, x_test, t_test,
                  epochs=max_epochs, mini_batch_size=100,
                  optimizer='Adam', optimizer_param={'lr': 0.001},
                  evaluate_sample_num_per_epoch=1000)
trainer.train()   
network.save_params("params.pkl")
print("Saved Network Parameters!")       
markers = {'train': 'o', 'test': 's'}
x = np.arange(max_epochs)
plt.plot(x, trainer.train_acc_list, marker='o', label='train', markevery=2)
plt.plot(x, trainer.test_acc_list, marker='s', label='test', markevery=2)
plt.xlabel("epochs")
plt.ylabel("accuracy")
plt.ylim(0, 1.0)
plt.legend(loc='lower right')
plt.show() 
            
            
            
            
            
            
            
        
        