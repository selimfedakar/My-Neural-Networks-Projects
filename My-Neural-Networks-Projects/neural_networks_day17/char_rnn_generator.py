import torch
import torch.nn as nn
import torch.nn.functional as F


# My Goal: Teach the model to predict the next character.
# "I realized that the network is actually a long pipeline of memory."

class CharRNN(nn.Module):
    def __init__(self, vocab_size, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size

        # 1. Embedding Layer: Character index -> Vector
        self.encoder = nn.Embedding(vocab_size, hidden_size)

        # 2. RNN Cell Logic: Combining Input + Previous Hidden
        # "I combine input and hidden state into a single linear transformation"
        self.rnn_cell = nn.Linear(hidden_size + hidden_size, hidden_size)

        # 3. Output Layer: Hidden state -> Character Probability
        self.decoder = nn.Linear(hidden_size, vocab_size)
        self.tanh = nn.Tanh()  # "The soul of the neural network"

    def forward(self, input_idx, hidden):
        # Character to vector
        embedded = self.encoder(input_idx)

        # "I concatenate the current input with the previous memory"
        combined = torch.cat((embedded, hidden), 1)

        # Update hidden state (Memory)
        hidden = self.tanh(self.rnn_cell(combined))

        # Predict next character (Logits)
        output = self.decoder(hidden)

        return output, hidden

    def init_hidden(self):
        # "I need to ensure my neurons are equally unsure at the start."
        return torch.zeros(1, self.hidden_size)


# --- THE SAMPLING TEST ---
vocab = " .abcdefghijklmnopqrstuvwxyz"  # My tiny universe
model = CharRNN(len(vocab), 128)

# Start with a dummy character 'a'
input_char = torch.tensor([vocab.index('a')])
hidden = model.init_hidden()

# One step forward
logits, next_hidden = model(input_char, hidden)

print(f"Logits shape: {logits.shape} (One prediction for each char in vocab)")
print("I've successfully set up the character-level prediction pipeline!")