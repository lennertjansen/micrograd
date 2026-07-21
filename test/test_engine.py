import torch
from micrograd.engine import Value

def test_sanity_check():

    # with micrograd
    x = Value(-8.0)
    z = 5 * x + 1 - x
    q = z.relu() + z * x
    h = (z * z).tanh()
    y = h + q + z + x * q
    y.backward()
    x_micrograd, y_micrograd = x, y

    # with pytorch
    x = torch.Tensor([-8.0]).double()
    x.requires_grad = True
    z = 5 * x + 1 - x
    q = z.relu() + z * x
    h = (z * z).tanh()
    y = h + q + z + x * q
    y.backward()
    x_pytorch, y_pytorch = x, y

    # forward pass
    assert y_micrograd.data == y_pytorch.data.item(), "forward passes disagree"

    # backward pass
    assert x_micrograd.grad == x_pytorch.grad.item(), "backward passes disagree"

def test_more_ops():

    # with micrograd
    a = Value(-4.0)
    b = Value(2.0)
    c = a + b
    d = a * b + b**3
    c += c + 1
    c += 1 + c + (-a)
    d += d * 2 + (b + a).tanh()
    d += 3 * d + (b - a).relu()
    e = c - d
    f = e**2
    g = f / 2.0
    g += 10.0 / f
    g.backward()
    a_micrograd, b_micrograd, g_micrograd = a, b, g

    # with pytorch
    a = torch.Tensor([-4.0]).double()
    b = torch.Tensor([2.0]).double()
    a.requires_grad = True
    b.requires_grad = True
    c = a + b
    d = a * b + b**3
    c = c + c + 1
    c = c + 1 + c + (-a)
    d = d + d * 2 + (b + a).tanh()
    d = d + 3 * d + (b - a).relu()
    e = c - d
    f = e**2
    g = f / 2.0
    g = g + 10.0 / f
    g.backward()
    a_pytorch, b_pytorch, g_pytorch = a, b, g

    tol = 1e-6 # tolerance

    # forward pass
    assert abs(g_micrograd.data - g_pytorch.data.item()) < tol, "forward passes disagree"
    
    # backward pass
    assert abs(a_micrograd.grad - a_pytorch.grad.item()) < tol, "backward passes disagree"
    assert abs(b_micrograd.grad - b_pytorch.grad.item()) < tol, "backward passes disagree"