# Neural Networks General — Foundation Reference

This folder is the starting point of the repository. It contains the core building blocks that everything else builds on.

## Contents

| File | Description |
|------|-------------|
| `karpathy_nn_foundation.py` | Reference guide based on Andrej Karpathy's micrograd lecture |
| `micrograd_lite.py` | Original scalar autograd engine (study copy) |
| `micrograd_litec.py` | Refactored version — cleaner ops, `__repr__`, `math.tanh` |

## What is this?

`karpathy_nn_foundation.py` is a personal reference file distilled from Andrej Karpathy's [micrograd](https://github.com/karpathy/micrograd) lecture and notebook. The code structure, pedagogy, and core ideas belong entirely to him (MIT License). It is kept here as a permanent mental model — the clearest possible explanation of how neural networks actually work at the lowest level.

**Six ideas this file makes concrete:**

1. A derivative is just the slope of a function at a point — how much the output changes when you nudge the input
2. Every operation (add, mul, tanh) has a simple local gradient rule
3. Chain rule connects those local rules from the output all the way back to the inputs
4. Topological sort ensures backward order is correct — children before parents, reversed
5. A neural network is just a big mathematical expression — loss is the scalar we differentiate
6. Gradient descent: `weight -= lr * weight.grad`, repeated until loss is small

## Why it starts here

The rest of this repository builds directly on these six ideas. Every day's code — from bigram models to batch normalization to RNNs — is an extension of what `Value.backward()` does in 15 lines.

---

*Code and pedagogy: Andrej Karpathy — [Neural Networks: Zero to Hero](https://github.com/karpathy/nn-zero-to-hero)*
