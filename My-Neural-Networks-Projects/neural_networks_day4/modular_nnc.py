import random
import math


class Value:
    """
    Scalar autograd engine — included here so this file runs standalone.
    'Micrograd is a scalar valued engine': we process one number at a time,
    but chain them together to build arbitrarily deep computation graphs.
    """

    def __init__(self, data, _children=(), _op='', label=''):
        self.data = data
        self.grad = 0.0
        self._backward = lambda: None
        self._prev = set(_children)  # Connective tissue of the computation graph
        self._op = _op
        self.label = label

    def __repr__(self):
        return f"Value(data={self.data:.4f}, grad={self.grad:.4f})"

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), '+')

        def _backward():
            self.grad += out.grad
            other.grad += out.grad

        out._backward = _backward
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), '*')

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad

        out._backward = _backward
        return out

    def __pow__(self, other):
        assert isinstance(other, (int, float))
        out = Value(self.data ** other, (self,), f'**{other}')

        def _backward():
            self.grad += (other * self.data ** (other - 1)) * out.grad

        out._backward = _backward
        return out

    def __radd__(self, other): return self + other
    def __rmul__(self, other): return self * other
    def __sub__(self, other):  return self + (other * -1)
    def __rsub__(self, other): return other + (self * -1)

    def tanh(self):
        # Squashes values into (-1, 1) — using math.tanh avoids computing exp(2x) twice
        t = math.tanh(self.data)
        out = Value(t, (self,), 'tanh')

        def _backward():
            self.grad += (1 - t ** 2) * out.grad

        out._backward = _backward
        return out

    def backward(self):
        topo = []
        visited = set()

        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)

        build_topo(self)
        self.grad = 1.0
        for node in reversed(topo):
            node._backward()


# --- MODULAR NN COMPONENTS ---

class Module:
    """Base class for all NN modules, as outlined in study notes."""

    def zero_grad(self):
        for p in self.parameters():
            p.grad = 0.0

    def parameters(self):
        return []


class Neuron(Module):
    """A single neuron: computes w·x + b, then squashes via tanh."""

    def __init__(self, nin):
        self.w = [Value(random.uniform(-1, 1)) for _ in range(nin)]
        self.b = Value(random.uniform(-1, 1))

    def __call__(self, x):
        act = sum((wi * xi for wi, xi in zip(self.w, x)), self.b)
        return act.tanh()

    def parameters(self):
        return self.w + [self.b]


class Layer(Module):
    """A collection of neurons — each sees the same input, produces independent outputs."""

    def __init__(self, nin, nout):
        self.neurons = [Neuron(nin) for _ in range(nout)]

    def __call__(self, x):
        outs = [n(x) for n in self.neurons]
        return outs[0] if len(outs) == 1 else outs

    def parameters(self):
        return [p for neuron in self.neurons for p in neuron.parameters()]


class MLP(Module):
    """The full Multi-Layer Perceptron — layers chained so each feeds the next."""

    def __init__(self, nin, nouts):
        sz = [nin] + nouts
        self.layers = [Layer(sz[i], sz[i + 1]) for i in range(len(nouts))]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        # Flat list across all layers — one call gives every learnable parameter
        return [p for layer in self.layers for p in layer.parameters()]


if __name__ == "__main__":
    # --- GRADIENT DESCENT (THE LEARNING) ---
    xs = [
        [2.0,  3.0, -1.0],
        [3.0, -1.0,  0.5],
        [0.5,  1.0,  1.0],
        [1.0,  1.0, -1.0],
    ]
    ys = [1.0, -1.0, -1.0, 1.0]

    n = MLP(3, [4, 4, 1])

    # "Gradient Descent is a loop" — forward, backward, nudge, repeat
    for step in range(100):

        # 1. Forward Pass
        ypred = [n(x) for x in xs]
        loss = sum(((yout - ygt) ** 2 for ygt, yout in zip(ys, ypred)), Value(0.0))

        # 2. Backward Pass (The Magic Line)
        n.zero_grad()
        loss.backward()

        # 3. The Nudge (Optimization) — step opposite to the gradient
        for p in n.parameters():
            p.data -= 0.05 * p.grad

        if step % 20 == 0:
            print(f"Step {step} | Loss: {loss.data:.4f}")

    print(f"Final predictions after 100 steps: {[round(p.data, 3) for p in ypred]}")
