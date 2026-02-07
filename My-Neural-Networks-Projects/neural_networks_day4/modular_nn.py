import random
import math

class Value:
    """ I implemented this to handle the core autograd and arithmetic logic. """
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
        assert isinstance(other, (int, float))
        out = Value(self.data**other, (self,), f'**{other}')
        def _backward():
            self.grad += (other * self.data**(other-1)) * out.grad
        out._backward = _backward
        return out

    def __radd__(self, other): return self + other
    def __rmul__(self, other): return self * other
    def __sub__(self, other): return self + (other * -1)
    def __rsub__(self, other): return other + (self * -1)

    def tanh(self):
        x = self.data
        t = (math.exp(2*x) - 1)/(math.exp(2*x) + 1)
        out = Value(t, (self, ), 'tanh')
        def _backward():
            self.grad += (1 - t**2) * out.grad
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
    """ Base class for all NN modules, as I outlined in my notes. """
    def zero_grad(self):
        for p in self.parameters():
            p.grad = 0
    def parameters(self): return []

class Neuron(Module):
    """ A single neuron implementing w*x + b. """
    def __init__(self, nin):
        self.w = [Value(random.uniform(-1,1)) for _ in range(nin)]
        self.b = Value(random.uniform(-1,1))
    def __call__(self, x):
        act = sum((wi*xi for wi, xi in zip(self.w, x)), self.b)
        return act.tanh()
    def parameters(self):
        # I return the weights list plus the bias
        return self.w + [self.b]

class Layer(Module):
    """ A collection of neurons. """
    def __init__(self, nin, nout):
        self.neurons = [Neuron(nin) for _ in range(nout)]
    def __call__(self, x):
        outs = [n(x) for n in self.neurons]
        return outs[0] if len(outs) == 1 else outs
    def parameters(self):
        # I collect parameters from every neuron in this layer
        return [p for neuron in self.neurons for p in neuron.parameters()]

class MLP(Module):
    """ The full Multi-Layer Perceptron. """
    def __init__(self, nin, nouts):
        sz = [nin] + nouts
        self.layers = [Layer(sz[i], sz[i+1]) for i in range(len(nouts))]
    def __call__(self, x):
        for layer in self.layers: x = layer(x)
        return x
    def parameters(self):
        # I collect parameters from all layers to create a single flat list
        return [p for layer in self.layers for p in layer.parameters()]

# --- GRADIENT DESCENT (THE LEARNING) ---

# Inputs and Targets
xs = [[2.0, 3.0, -1.0], [3.0, -1.0, 0.5], [0.5, 1.0, 1.0], [1.0, 1.0, -1.0]]
ys = [1.0, -1.0, -1.0, 1.0]

n = MLP(3, [4, 4, 1])

# As I noted: "Gradient Descent is a loop"
for step in range(100):
    # 1. Forward Pass
    ypred = [n(x) for x in xs]
    loss = sum(((yout - ygt)**2 for ygt, yout in zip(ys, ypred)), Value(0.0))

    # 2. Backward Pass (The Magic Line)
    n.zero_grad()
    loss.backward()

    # 3. The Nudge (Optimization)
    # Using the flat list from my helper function
    for p in n.parameters():
        p.data += -0.05 * p.grad

    if step % 20 == 0:
        print(f"Step {step} | Loss: {loss.data:.4f}")

print(f"Final predictions after 100 steps: {[round(p.data, 3) for p in ypred]}")