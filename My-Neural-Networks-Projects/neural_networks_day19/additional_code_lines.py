#1. The "Base" Modules (Linear & Tanh)
#To build a Sequential model, every layer must have the same interface.
#They all need a __call__ method to process data and a parameters() method to return their weights.

class Linear:
    def __init__(self, fan_in, fan_out, bias=True):
        # Kaiming Initialization (Standard for Tanh/ReLU)
        self.weight = torch.randn((fan_in, fan_out)) / fan_in ** 0.5
        self.bias = torch.zeros(fan_out) if bias else None

    def __call__(self, x):
        # Matrix Multiplication: y = xW + b
        self.out = x @ self.weight
        if self.bias is not None:
            self.out += self.bias
        return self.out

    def parameters(self):
        return [self.weight] + ([] if self.bias is None else [self.bias])


class Tanh:
    def __call__(self, x):
        self.out = torch.tanh(x)
        return self.out

    def parameters(self):
        return []

    #The Hierarchical Trick: FlattenConsecutive
#This is the piece that solves the "Hierarchical Scheme" you asked about.
#This replaces the standard Flatten layer.

class FlattenConsecutive:
    def __init__(self, n):
        self.n = n  # Number of consecutive elements to fuse (usually 2)

    def __call__(self, x):
        # x shape: (B, T, C) -> (Batch, Time, Channels)
        B, T, C = x.shape
        # We reshape to (B, T//n, C*n)
        # Example: (32, 8, 10) becomes (32, 4, 20)
        x = x.view(B, T // self.n, C * self.n)

        # If the middle dimension becomes 1, we 'squeeze' it out
        # (This avoids that "spurious dimension 1" Karpathy mentioned)
        if x.shape[1] == 1:
            x = x.squeeze(1)

        self.out = x
        return self.out

    def parameters(self):
        return []

    #The "Sequential" Container
#This is the "Pipeline." It allows you to stack layers like LEGO bricks.
class Sequential:
    def __init__(self, layers):
        self.layers = layers

    def __call__(self, x):
        # The data 'flows' through each layer in the list
        for layer in self.layers:
            x = layer(x)
        self.out = x
        return self.out

    def parameters(self):
        # Collect all parameters from all layers into one list
        return [p for layer in self.layers for p in layer.parameters()]

    #The Final Model Construction (Minute 48:45)
#This is where he puts it all together to create the "Tree" structure.
#Notice how he alternates between Linear and FlattenConsecutive to slowly fuse the information.

# Ahmet Selim (me) and everyone, notice how the input dimension doubles !!!!
# each time we fuse 2 characters together (10 -> 20 -> 40)
n_embd = 10
n_hidden = 68 # Hyperparameter for width

model = Sequential([
    Embedding(vocab_size, n_embd),
    # First Fusion: 8 chars -> 4 pairs
    FlattenConsecutive(2), Linear(n_embd * 2, n_hidden, bias=False), BatchNorm1d(n_hidden), Tanh(),
    # Second Fusion: 4 pairs -> 2 quads
    FlattenConsecutive(2), Linear(n_hidden * 2, n_hidden, bias=False), BatchNorm1d(n_hidden), Tanh(),
    # Third Fusion: 2 quads -> 1 octet (Final hidden state)
    FlattenConsecutive(2), Linear(n_hidden * 2, n_hidden, bias=False), BatchNorm1d(n_hidden), Tanh(),
    # Final Output Layer
    Linear(n_hidden, vocab_size),
])

