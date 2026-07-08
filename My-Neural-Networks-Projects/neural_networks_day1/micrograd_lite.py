import math


class Value:
    """
    Represents a scalar value and its gradient for automatic differentiation.
    Based on handwritten study notes on Micrograd and Neural Networks.
    """

    def __init__(self, data, _children=(), _op='', label=''):
        self.data = data
        self.grad = 0.0
        self._backward = lambda: None
        self._prev = set(_children)  # Connective tissue of the computation graph
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

    def tanh(self):
        """ Squashes values for non-linearity as seen in neuron layers. """
        x = self.data
        t = (math.exp(2 * x) - 1) / (math.exp(2 * x) + 1)
        out = Value(t, (self,), 'tanh')

        def _backward():
            self.grad += (1 - t ** 2) * out.grad

        out._backward = _backward
        return out

    def backward(self):
        """ Triggers backpropagation through the topological sort of the graph. """
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

            # --- Verification with handwritten notes ---
            # Initializing variables from your study notes
            
a = Value(2.0, label='a')
b = Value(-3.0, label='b')
c = Value(10.0, label='c')

# Forward pass: (a * b + c) and then tanh activation
e = a * b;
e.label = 'e'
d = e + c;
d.label = 'd'
L = d.tanh();
L.label = 'L'

# Backpropagation: The magic happens here!
L.backward()

print("--- Neural Network Day 1 Results ---")
print(f"Final Output (L) data: {L.data:.4f}")
print(f"Gradient dL/da: {a.grad:.4f}")
print(f"Gradient dL/db: {b.grad:.4f}")
print(f"Gradient dL/dc: {c.grad:.4f}")