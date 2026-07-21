import math

class Value:
    """TODO: Describe Value object in your own words."""

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
        
        out = Value(data=self.data+other.data, _children=(self, other), _op='+')
        
        def _backward():
            self.grad += out.grad # remember it's out.grad * other.grad with other.grad = 1, because of the chain rule
            other.grad += out.grad
        out._backward = _backward

        return out

    def __mul__(self, other):
        "Multiplication"

        out = Value(data = self.data*other.data, _children=(self, other), _op='*')

        def _backward():
            self.grad += out.grad * other.data
            other.grad += out.grad * self.data
        out._backward = _backward
        
        return out

    def tanh(self):
        x = self.data
        t = (math.exp(2*x) - 1) / (math.exp(2*x) + 1)
        out = Value(t, (self, ), _op='tanh')

        def _backward():
            self.grad += (1 - t**2) * out.grad
        out._backward = _backward


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