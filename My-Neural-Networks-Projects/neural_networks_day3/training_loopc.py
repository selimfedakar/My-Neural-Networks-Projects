import random
import math


class Value:
    """
    Scalar autograd engine — rebuilt here so this file runs standalone.
    As I noted: 'Micrograd is a scalar valued engine' — we process numbers one by one.
    Each day adds new operations as the training loop demands them.
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
        assert isinstance(other, (int, float)), "Only supporting int/float powers for now"
        out = Value(self.data ** other, (self,), f'**{other}')

        def _backward():
            self.grad += (other * self.data ** (other - 1)) * out.grad

        out._backward = _backward
        return out

    # --- Critical fixes for arithmetic & sum() compatibility ---

    def __radd__(self, other):  # Handles: int/float + Value
        return self + other

    def __rmul__(self, other):  # Handles: int/float * Value
        return self * other

    def __sub__(self, other):   # Handles: Value - Value
        return self + (other * -1)

    def __rsub__(self, other):  # Handles: int/float - Value
        return other + (self * -1)

    # -----------------------------------------------------------

    def tanh(self):
        # Squashes values into (-1, 1) for non-linearity in neuron layers
        # Using math.tanh directly — avoids computing exp(2x) twice
        t = math.tanh(self.data)
        out = Value(t, (self,), 'tanh')

        def _backward():
            # Derivative of tanh: (1 - tanh²(x))
            self.grad += (1 - t ** 2) * out.grad

        out._backward = _backward
        return out

    def backward(self):
        """Triggers backpropagation through the topological sort of the graph."""
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


# --- Neural Network Modules ---

class Module:
    def zero_grad(self):
        for p in self.parameters():
            p.grad = 0.0

    def parameters(self):
        return []


class Neuron(Module):
    def __init__(self, nin):
        self.w = [Value(random.uniform(-1, 1)) for _ in range(nin)]
        self.b = Value(random.uniform(-1, 1))

    def __call__(self, x):
        # Using self.b as the starting value for sum to avoid integer 0 issues
        act = sum((wi * xi for wi, xi in zip(self.w, x)), self.b)
        return act.tanh()

    def parameters(self):
        return self.w + [self.b]


class Layer(Module):
    def __init__(self, nin, nout):
        self.neurons = [Neuron(nin) for _ in range(nout)]

    def __call__(self, x):
        outs = [n(x) for n in self.neurons]
        return outs[0] if len(outs) == 1 else outs

    def parameters(self):
        return [p for n in self.neurons for p in n.parameters()]


class MLP(Module):
    def __init__(self, nin, nouts):
        sz = [nin] + nouts
        self.layers = [Layer(sz[i], sz[i + 1]) for i in range(len(nouts))]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]


if __name__ == "__main__":
    # --- THE TRAINING LOOP ---
    xs = [
        [2.0,  3.0, -1.0],
        [3.0, -1.0,  0.5],
        [0.5,  1.0,  1.0],
        [1.0,  1.0, -1.0],
    ]
    ys = [1.0, -1.0, -1.0, 1.0]

    n = MLP(3, [4, 4, 1])

    print("Starting Training Loop...")
    for k in range(50):

        # 1. Predict (Forward Pass)
        ypred = [n(x) for x in xs]

        # 2. Check Error (Calculate Loss)
        # Corrected loss line: starts with Value(0.0) — not integer 0
        loss = sum(((yout - ygt) ** 2 for ygt, yout in zip(ys, ypred)), Value(0.0))

        # 3. Find Gradients (Backpropagation)
        n.zero_grad()
        loss.backward()

        # 4. Update Weights — step in the opposite direction of the gradient
        learning_rate = 0.1
        for p in n.parameters():
            p.data -= learning_rate * p.grad

        if k % 10 == 0:
            print(f"Step {k} | Loss: {loss.data:.4f}")

    print("-" * 30)
    print(f"Final Predictions: {[round(p.data, 3) for p in ypred]}")

    # --- Why __radd__ was necessary (discovered during this session) ---
    #
    # When Python evaluates sum([Value(1), Value(2)]), it does this internally:
    #   result = 0          (integer — Python's default start for sum())
    #   result = result + Value(1)  →  CRASH: int doesn't know how to add a Value
    #
    # Two fixes applied:
    #   1. Pass Value(0.0) as the start argument to sum() — sidesteps the integer entirely
    #   2. Implement __radd__ — makes Value "smart" enough to handle being on either
    #      side of the + sign, so 0 + Value(x) falls back to Value.__radd__ and works.
    #
    # Now it's perfectly working.
