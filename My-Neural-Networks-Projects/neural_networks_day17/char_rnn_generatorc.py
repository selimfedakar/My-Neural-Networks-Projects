import torch
import torch.nn as nn

# Goal: teach the model to predict the next character.
# "The network is actually a long pipeline of memory."


class CharRNN(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size

        # 1. Embedding Layer: character index → dense vector
        self.encoder = nn.Embedding(vocab_size, embed_size)

        # 2. RNN Cell: combines current input with previous hidden state
        # "I combine input and hidden state into a single linear transformation"
        self.rnn_cell = nn.Linear(embed_size + hidden_size, hidden_size)

        # 3. Output Layer: hidden state → character logits
        self.decoder = nn.Linear(hidden_size, vocab_size)
        self.tanh = nn.Tanh()  # "The soul of the neural network"

    def forward(self, input_idx, hidden):
        embedded = self.encoder(input_idx)

        # "Concatenate the current input with the previous memory"
        combined = torch.cat((embedded, hidden), dim=1)

        # Update hidden state — new memory from old memory + new input
        hidden = self.tanh(self.rnn_cell(combined))

        # Predict next character
        output = self.decoder(hidden)
        return output, hidden

    def init_hidden(self, batch_size=1):
        # "Neurons equally unsure at the start" — zero hidden state
        return torch.zeros(batch_size, self.hidden_size)


if __name__ == "__main__":
    # --- THE SAMPLING TEST ---
    vocab     = " .abcdefghijklmnopqrstuvwxyz"
    model     = CharRNN(vocab_size=len(vocab), embed_size=32, hidden_size=128)

    input_char = torch.tensor([vocab.index('a')])
    hidden     = model.init_hidden()

    logits, next_hidden = model(input_char, hidden)

    print(f"Logits shape: {logits.shape}  (one prediction per character in vocab)")
    print("Successfully set up the character-level prediction pipeline!")
