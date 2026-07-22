# micrograd

My from-scratch implementation of [Karpathy's micrograd](https://github.com/karpathy/micrograd): a tiny scalar-valued autograd engine (`micrograd/engine.py`) and a small neural net library on top of it (`micrograd/nn.py`) with a PyTorch-like API. Backprop runs over a DAG of scalar `Value`s.

Built by following ["The spelled-out intro to neural networks and backpropagation"](https://www.youtube.com/watch?v=VMj-3S1tku0), with `micrograd_from_scratch.ipynb` as the scratchpad along the way.

<!-- TODO(lennert): a line or two on what I took away from building this -->

## Installation

Everything runs through [uv](https://docs.astral.sh/uv/) (Python 3.11+):

```bash
uv sync
```

## Example usage

```python
from micrograd.engine import Value

a = Value(-4.0)
b = Value(2.0)
c = a + b
d = a * b + b**3
e = (d - c).relu() + c.tanh()
f = e**2 / 2.0
f.backward()
print(f'{f.data:.4f}')  # 0.5366, the outcome of the forward pass
print(f'{a.grad:.4f}')  # 1.1092, df/da
print(f'{b.grad:.4f}')  # 7.3250, df/db
```

## Training a neural net

`micrograd/nn.py` provides `Neuron`, `Layer`, and `MLP` modules:

```python
from micrograd.nn import MLP

model = MLP(3, [4, 4, 1])  # 3 inputs, two hidden layers of 4, 1 output
y = model([2.0, 3.0, -1.0])
y.backward()
# gradient descent: p.data -= lr * p.grad for every p in model.parameters(),
# then model.zero_grad() and repeat
```

The full training loop (MSE loss, manual gradient descent) is in the notebook:

```bash
uv run jupyter lab micrograd_from_scratch.ipynb
```

## Running tests

The tests cross-check forward and backward passes against PyTorch:

```bash
uv run pytest
```

## License

MIT
