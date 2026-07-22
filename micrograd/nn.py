import random

from micrograd.engine import Value

class Module:

    def zero_grad(self):
        for p in self.parameters():
            p.grad = 0
    
    def parameters(self):
        return []

class Neuron(Module):

    def __init__(self, nin, nonlin=True):
        self.weights = [Value(random.uniform(-1, 1)) for _ in range(nin)]
        self.bias = Value(random.uniform(-1, 1))
        self.nonlin = nonlin

    def __call__(self, x):
        act = sum([wi * xi for wi, xi in zip(self.weights, x)], self.bias)
        return act.relu() if self.nonlin else act

    def parameters(self):
        return self.weights + [self.bias]

class Layer(Module):
    
    def __init__(self, nin, nout, **kwargs):
        self.neurons = [Neuron(nin, **kwargs) for _ in range(nout)]

    def __call__(self, x):
        out = [n(x) for n in self.neurons]
        return out[0] if len(out) == 1 else out

    def parameters(self):
        return [p for neuron in self.neurons for p in neuron.parameters()]

        
class MLP(Module):
    
    def __init__(self, nin: int, nouts: list[int]):
        dims = [nin] + nouts
        self.layers = [Layer(dims[i], dims[i + 1], nonlin=i!=len(nouts)-1) for i in range(len(nouts))]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]