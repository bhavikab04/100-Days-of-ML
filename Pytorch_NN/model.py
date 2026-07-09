import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module): #we are defining class MLP and it inherits form nn.Module, it's a subclass and it extends the parent class
    def __init__(self): #constructor of the class; note that self is the object itself, it's same as "this" pointer in Java
        super().__init__() #parent's constructor being called; sets up all the internal bookkeeping that makes parameter-tracking work
        self.fc0 = nn.Linear(64,64) #replaces w0+b0; note that nn.LInear is a class, so here we are making fc0 an object of that class
        self.fc1 = nn.Linear(64,32) #replaces w1 + b1
        self.fc2 = nn.Linear(32,10) #replaces w2+b2
    def forward(self, x): #it is a method form the parent class, and the __call__ looks for the forward method
        a1 = F.relu(self.fc0(x)) #call fc0 object on x (triggers its __call__), then relu; basically even tho fc0 is an object, it is a callable object as the original class used python's __call__ function, which enables the object to be callable. 
        # ...actually runs fc0.__call__(x), which runs fc0.forward(x) internally
        a2 = F.relu(self.fc1(a1))
        z2 = self.fc2(a2) #raw digits with no softmax
        return z2


