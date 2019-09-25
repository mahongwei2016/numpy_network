import numpy as np
class Mullayer:
    def __init__(self):
        self.x=None
        self.y=None
        
    def forward(self,x,y):
        self.x=x
        self.y=y
        out=x*y
        return out
    def backward(self,dout):
        dx=dout*self.y
        dy=dout*self.x
        return dx,dy
    
'''
apple=100
apple_num=2
tax=1.1
mul_apple_layer=Mullayer()
mul_tax_layer=Mullayer()
#forward
apple_prick=mul_apple_layer.forward(apple, apple_num)
price=mul_tax_layer.forward(apple_prick,tax)
print(price)
#backward
dprice=1
dapple_prick,dtax=mul_tax_layer.backward(dprice)
print(dapple_prick)
print(dtax)
dapple,dapple_num=mul_apple_layer.backward(dapple_prick)
print(dapple)
print(dapple_num)
'''