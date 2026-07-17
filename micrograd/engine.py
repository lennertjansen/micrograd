class Value:

    def __init__(self, data, children = ()):
        self.data = data
        self.grad = 0
        self.children = children
        self._backward = lambda: None

    def __add__(self, other):
        "Addition"
        
        out = Value(data=self.data+other.data, children=(self, other))
        
        def _backward():
            self.grad += out.grad # remember it's out.grad * other.grad with other.grad = 1, because of the chain rule
            other.grad += out.grad
        out._backward = _backward

        return out

    def __mul__(self, other):
        "Multiplication"

        out = Value(data = self.data*other.data, children=(self, other))

        def _backward():
            self.grad += out.grad * other.data
            other.grad += out.grad * self.data
        out._backward = _backward
        
        return out