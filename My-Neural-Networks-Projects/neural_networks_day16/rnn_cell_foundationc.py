import torch
import torch.nn as nn

# An MLP processes one fixed-size input and has no memory of what came before.
# An RNN solves this: it carries a 'hidden state' — a summary of all previous inputs.
# This hidden state is the basic 'Memory' unit of an RNN.


class SimpleRNNCell(nn.Module):
    """
    One RNN step: takes the current input and the previous hidden state,
    combines them, and produces a new hidden state (updated memory).

    hidden_t = tanh( Linear([ input_t | hidden_{t-1} ]) )
    """

    def __init__(self, input_size, hidden_size):
        super().__init__()
        # i2h: maps concatenated [input | hidden] → new hidden
        # input_size + hidden_size because we concatenate before projecting
        self.i2h  = nn.Linear(input_size + hidden_size, hidden_size)
        self.tanh = nn.Tanh()

    def forward(self, input, hidden):
        # Concatenate current input and previous memory along the feature dimension
        combined = torch.cat((input, hidden), dim=1)
        # The new memory is a non-linear mix of what just arrived and what was remembered
        hidden = self.tanh(self.i2h(combined))
        return hidden


if __name__ == "__main__":
    input_size  = 10
    hidden_size = 20
    batch_size  = 4
    seq_len     = 5  # Process a sequence of 5 time steps

    cell   = SimpleRNNCell(input_size, hidden_size)
    hidden = torch.zeros(batch_size, hidden_size)  # Memory starts empty

    print(f"Processing a sequence of {seq_len} time steps...")
    for t in range(seq_len):
        x      = torch.randn(batch_size, input_size)  # One token per batch element
        hidden = cell(x, hidden)
        print(f"  Step {t + 1} | Hidden state shape: {hidden.shape}")

    print("-" * 30)
    print(f"Final hidden state (memory after {seq_len} steps): {hidden.shape}")
