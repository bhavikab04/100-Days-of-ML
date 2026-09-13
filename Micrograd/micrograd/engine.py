import math
class Value:
    def __init__(self, data, _children=(), _op=""):
        self.data = data
        self.grad = 0.0
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op

    def __repr__(self):
        return f"Value(data={self.data}, grad={self.grad})"

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        c = Value(self.data + other.data, (self, other), "+")
        def _backward(): # c = a + b so dc/da = 1 and dc/db = 1; in general parent.grad = dL/dparent = output.grad (dL/doutput) * doutput/dparent --> here parent is a and b for output c
            #so parent.grad = c.grad * 1 as dc/da and dc/db = 1 as it's just addition; also we start with self.grad = 0.0 and gradients accumulate; like partial gradients from multiple output functions need to be added up for the total gradient 
            self.grad += c.grad
            other.grad += c.grad
        c._backward = _backward
        return c

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        c = Value(self.data * other.data, (self, other), "*")
        def _backward():# c= a*b so dc/da = b and dc/db = a
            self.grad += other.data * c.grad
            other.grad += self.data * c.grad
        c._backward = _backward
        return c    

    def __neg__(self):
        c = Value(-1 * self.data, {self}, "-")
        def _backward(): # c = -a dc/da 
            self.grad -= c.grad
        c._backward = _backward
        return c

    def __sub__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        c = Value(self.data - other.data, (self, other), "-")
        def _backward(): # c = a - b so dc/da = 1 and dc/db = -1
            self.grad += c.grad
            other.grad -= c.grad
        c._backward = _backward
        return c

    def __pow__(self, other): #other is only allowed ot be int here, or some number only not a variable
        c = Value(self.data ** other, {self}, "^")
        def _backward(): # c = a ^ n; dc/da = n * a ^ (n-1); a.grad = c.grad * dc/da
            self.grad += c.grad * (other*(self.data**(other-1))) # ** is the python operator for ^; ^ in python is bitwise or operator
        c._backward = _backward
        return c

    def __truediv__(self, other):
        return self.__mul__(other.__pow__(-1))     

    def tanh(self):
        c = Value(math.tanh(self.data), {self}, "tanh")
        def _backward(): # c = tanh(a); 
            self.grad += (1-(c.data ** 2)) * (c.grad)
        c._backward = _backward
        return c

    def relu(self):
        c = Value(max(0,self.data), {self}, "relu")
        def _backward(): # c = tanh(a); 
            self.grad += (1 if self.data> 0 else 0)* c.grad
        c._backward = _backward
        return c

    def backward(self):
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v) # will append inputs first as we wait for the loop above to end to call the append; we end up adding current child node only after the parent nodes ar eadded in the prev csalls
        build_topo(self)
        self.grad = 1.0
        for v in reversed(topo):#processing outputs first for backprop
            v._backward()    

