import math

class Value:
    """
    A scalar value in a computation graph that support automatic differentiation. 
    
    Each operation builds the graphs by recording its inputs, 
    and backward() walks that graph in reverse topological order applying the chain rule to fill in .grad.
    """

    def __init__(self, data, _children = (), _op=''):
        self.data = data
        self.grad = 0.0
        self._prev = set(_children)
        self._op = _op
        self._backward = lambda: None

    def __repr__(self):
        return f"Value(data={self.data})"

    def __add__(self, other):
        "Addition"
        other = other if isinstance(other, Value) else Value(other)
        out = Value(data=self.data+other.data, _children=(self, other), _op='+')
        
        def _backward():
            self.grad += out.grad # remember it's out.grad * other.grad with other.grad = 1, because of the chain rule
            other.grad += out.grad
        out._backward = _backward

        return out

    def __mul__(self, other):
        "Multiplication"

        other = other if isinstance(other, Value) else Value(other)
        out = Value(data = self.data*other.data, _children=(self, other), _op='*')

        def _backward():
            self.grad += out.grad * other.data
            other.grad += out.grad * self.data
        out._backward = _backward
        
        return out

    def __pow__(self, other):
        "Exponentiation"

        assert isinstance(other, (int, float)), "only supporting int or float exponents"

        out = Value(data=self.data**other, _children=(self, ), _op=f'**{other}')

        def _backward():
            self.grad += out.grad * other * self.data**(other - 1)
        out._backward = _backward

        return out

    def tanh(self):
        "Hyperbolic tangent."

        x = self.data
        # use the form whose exponent is <= 0 so exp() can't overflow for large |x|
        if x >= 0:
            e = math.exp(-2*x)
            t = (1 - e) / (1 + e)
        else:
            e = math.exp(2*x)
            t = (e - 1) / (e + 1)
        out = Value(data=t, _children=(self, ), _op='tanh')

        def _backward():
            self.grad += (1 - t**2) * out.grad
        out._backward = _backward

        return out

    def relu(self):
        "Rectifier linear unit."

        out = Value(data = (0 if self.data < 0 else self.data), _children=(self,), _op="ReLU")

        def _backward():
            self.grad += (out.data > 0) * out.grad
        out._backward = _backward

        return out


    def backward(self):
        topo = []
        visited = set()

        def topo_sort(node):
            "Topological sort"
            if node not in visited:
                visited.add(node)
                for child in node._prev:
                    topo_sort(child)
                topo.append(node)
        topo_sort(self)

        self.grad = 1.0 #seed: derivative of the output with respect to itself
        for node in reversed(topo):
            node._backward()
    
    def __neg__(self): # -self
        return self * -1

    def __radd__(self, other): # other + self
        return self + other

    def __sub__(self, other): # self - other
        return self + (-other)

    def __rsub__(self, other): # other - self
        return other + (-self)

    def __rmul__(self, other): # other * self
        return self * other

    def __truediv__(self, other): # self / other
        return self * other**-1

    def __rtruediv__(self, other): # other / self
        return other * self**-1