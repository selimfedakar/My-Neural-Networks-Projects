import random
import math


class Value:
    """
    I implemented this scalar engine to handle automatic differentiation.
    As I noted: 'Micrograd is a scalar valued engine' - we process numbers one by one.
    """

    def __init__(self, data, _children=(), _op='', label=''):
        self.data = data
        self.grad = 0.0
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op
        self.label = label

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), '+')

        def _backward():
            self.grad += 1.0 * out.grad
            other.grad += 1.0 * out.grad

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

    # --- CRITICAL FIXES FOR ARITHMETIC & SUM() ---

    def __radd__(self, other):  # Handles: int/float + Value
        return self + other

    def __rmul__(self, other):  # Handles: int/float * Value
        return self * other

    def __sub__(self, other):  # Handles: Value - Value
        return self + (other * -1)

    def __rsub__(self, other):  # Handles: int/float - Value
        return other + (self * -1)

    # ---------------------------------------------

    def tanh(self):
        x = self.data
        t = (math.exp(2 * x) - 1) / (math.exp(2 * x) + 1)
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

    def __repr__(self):
        return f"Value(data={self.data}, grad={self.grad})"


# --- Neural Network Modules ---

class Module:
    def zero_grad(self):
        for p in self.parameters():
            p.grad = 0

    def parameters(self): return []


class Neuron(Module):
    def __init__(self, nin):
        self.w = [Value(random.uniform(-1, 1)) for _ in range(nin)]
        self.b = Value(random.uniform(-1, 1))

    def __call__(self, x):
        # Using self.b as the starting value for sum to avoid integer 0 issues
        act = sum((wi * xi for wi, xi in zip(self.w, x)), self.b)
        return act.tanh()

    def parameters(self): return self.w + [self.b]


class Layer(Module):
    def __init__(self, nin, nout):
        self.neurons = [Neuron(nin) for _ in range(nout)]

    def __call__(self, x):
        outs = [n(x) for n in self.neurons]
        return outs[0] if len(outs) == 1 else outs

    def parameters(self): return [p for n in self.neurons for p in n.parameters()]


class MLP(Module):
    def __init__(self, nin, nouts):
        sz = [nin] + nouts
        self.layers = [Layer(sz[i], sz[i + 1]) for i in range(len(nouts))]

    def __call__(self, x):
        for layer in self.layers: x = layer(x)
        return x

    def parameters(self): return [p for layer in self.layers for p in layer.parameters()]


# --- THE TRAINING LOOP ---

xs = [
    [2.0, 3.0, -1.0],
    [3.0, -1.0, 0.5],
    [0.5, 1.0, 1.0],
    [1.0, 1.0, -1.0],
]
ys = [1.0, -1.0, -1.0, 1.0]

n = MLP(3, [4, 4, 1])

print("Starting Training Loop...")
for k in range(50):

    # 1. Predict (Forward Pass)
    ypred = [n(x) for x in xs]

    # 2. Check Error (Calculate Loss)
    # Corrected loss line: starts with Value(0.0)
    loss = sum(((yout - ygt) ** 2 for ygt, yout in zip(ys, ypred)), Value(0.0))

    # 3. Find Gradients (Backpropagation)
    n.zero_grad()
    loss.backward()

    # 4. Update Weights (Step 4 from my notes)
    learning_rate = 0.1
    for p in n.parameters():
        p.data += -learning_rate * p.grad

    if k % 10 == 0:
        print(f"Step {k} | Loss: {loss.data:.4f}")

print("-" * 30)
print(f"Final Predictions: {[round(p.data, 3) for p in ypred]}")

#I solved a exception and why did this happen?
#When I call sum([Value(1), Value(2)]), Python internally does this:
#result = 0 (integer)
#result = result + Value(1) ➡️ CRASH! An integer doesn't know how to add a Value.
#By adding __radd__, I am making my Value class "smart" enough to handle being on either side of the + sign.

#now its perfectly working 