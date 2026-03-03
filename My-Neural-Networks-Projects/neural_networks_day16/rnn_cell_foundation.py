import torch
import torch.nn as nn

# Initializing the basic 'Memory' unit of an RNN
class SimpleRNNCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        # Memory is updated by combining current input and previous hidden state
        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
        self.tanh = nn.Tanh()

    def forward(self, input, hidden):
        # Concatenate input and previous memory
        combined = torch.cat((input, hidden), 1)
        hidden = self.tanh(self.i2h(combined))
        return hidden