# Micrograd

My implementation of Karpathy's Micrograd: a tiny scalar-valued autograd engine and a small neural net library on top of it with a PyTorch-like API. Mostly in `micrograd_from_scratch.ipynb`.

## Setup

Uses [uv](https://docs.astral.sh/uv/) for dependency and environment management (Python 3.11+).

```bash
uv sync          # create the venv and install deps
```

## Usage

Open the notebook to walk through the engine and a small neural net:

```bash
uv run jupyter lab micrograd_from_scratch.ipynb
```

## References

* Karpathy's micrograd: https://github.com/karpathy/micrograd
* "The spelled-out intro to neural networks and backpropagation": https://www.youtube.com/watch?v=VMj-3S1tku0
