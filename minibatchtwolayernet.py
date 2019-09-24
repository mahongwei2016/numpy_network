import numpy as np
import matplotlib.pylab as plt
from dataset.mnist import load_mnist
from twolayernet import twolayernet
#from ch04.two_layer_net import TwoLayerNet

(x_train,t_train),(x_test,t_test)=load_mnist(normalize=True,one_hot_label=True)
train_loss_list=[]
iters_num=10000
train_size=x_train.shape[0]
batch_size=100
learning_rate=0.1
network=twolayernet(input_size=784,hidden_size=50,output_size=10)
#network=TwoLayerNet(input_size=784,hidden_size=50,output_size=10)
train_loss_list=[]
train_acc_list=[]
test_acc_list=[]
iter_per_epoch=max(train_size/batch_size,1)
for i in range(iters_num):
    batch_mask=np.random.choice(train_size,batch_size)
    x_batch=x_train[batch_mask]
    t_batch=t_train[batch_mask]
    #grad=network.numerical_gradient(x_batch, t_batch)
    grad=network.gradient(x_batch, t_batch)
    #for key in ('w1','b1','w2','b2'):
    for key in ('W1','b1','W2','b2'):
        network.params[key]-=learning_rate*grad[key]
        #print(key+":"+str(network.params[key]))
    loss=network.loss(x_batch, t_batch)
    #print(loss)
    train_loss_list.append(loss)
    if i % iter_per_epoch==0:
        train_acc=network.accuracy(x_train,t_train)
        test_acc=network.accuracy(x_test,t_test)
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)
        print("train acc,test acc"+str(train_acc)+","+str(test_acc))

markers={'train':'o','test':'s'}
x=np.arange(len(train_acc_list))
plt.plot(x,train_acc_list,label='train acc')
plt.plot(x,test_acc_list,label='test acc')
plt.xlabel("epochs")
plt.ylabel("accuracy")
plt.ylim(0,1.0)
plt.legend(loc='lower right')
plt.show()
