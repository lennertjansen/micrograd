import random

import torch
from micrograd.nn import MLP

def test_parameters_and_zero_grad():

    model = MLP(3, [4, 4, 1])

    # 4*(3+1) + 4*(4+1) + 1*(4+1) weights and biases
    assert len(model.parameters()) == 41, "wrong parameter count"

    y = model([1.0, 2.0, 3.0])
    y.backward()
    assert any(p.grad != 0 for p in model.parameters()), "backward left all grads zero"

    model.zero_grad()
    assert all(p.grad == 0 for p in model.parameters()), "zero_grad left nonzero grads"

def test_mlp_forward_backward():

    # with micrograd
    random.seed(1337)
    model = MLP(2, [3, 1])
    y = model([1.0, -2.0])
    y.backward()
    y_micrograd = y

    # with pytorch, using the same weights
    h = torch.tensor([1.0, -2.0]).double()
    Ws, bs = [], []
    for i, layer in enumerate(model.layers):
        W = torch.tensor([[w.data for w in n.weights] for n in layer.neurons]).double()
        b = torch.tensor([n.bias.data for n in layer.neurons]).double()
        W.requires_grad = True
        b.requires_grad = True
        Ws.append(W)
        bs.append(b)
        h = h @ W.T + b
        if i != len(model.layers) - 1:
            h = h.relu()
    y = h.squeeze()
    y.backward()
    y_pytorch = y

    tol = 1e-6 # tolerance

    # forward pass
    assert abs(y_micrograd.data - y_pytorch.data.item()) < tol, "forward passes disagree"

    # backward pass
    for layer, W, b in zip(model.layers, Ws, bs):
        for j, neuron in enumerate(layer.neurons):
            for k, w in enumerate(neuron.weights):
                assert abs(w.grad - W.grad[j, k].item()) < tol, "backward passes disagree"
            assert abs(neuron.bias.grad - b.grad[j].item()) < tol, "backward passes disagree"
