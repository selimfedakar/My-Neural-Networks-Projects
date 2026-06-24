import torch
import torch.nn as nn
import torch.nn.functional as F

# Goal: generate a sequence of characters using 'Temperature' to control creativity.
# "I divide logits by temperature to control the 'confidence' of the model."


# --- Minimal model and vocab — defined here so this file runs standalone ---
vocab = " .abcdefghijklmnopqrstuvwxyz"

class CharRNN(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.encoder  = nn.Embedding(vocab_size, embed_size)
        self.rnn_cell = nn.Linear(embed_size + hidden_size, hidden_size)
        self.decoder  = nn.Linear(hidden_size, vocab_size)
        self.tanh     = nn.Tanh()

    def forward(self, input_idx, hidden):
        embedded = self.encoder(input_idx)
        combined = torch.cat((embedded, hidden), dim=1)
        hidden   = self.tanh(self.rnn_cell(combined))
        output   = self.decoder(hidden)
        return output, hidden

    def init_hidden(self, batch_size=1):
        return torch.zeros(batch_size, self.hidden_size)
# ---------------------------------------------------------------------------


def sample_from_model(model, vocab, start_str, length, temperature=1.0):
    model.eval()  # "I am now in evaluation mode"
    chars  = list(start_str)
    hidden = model.init_hidden()

    # 1. Warm up the hidden state with the starting string
    for char in start_str[:-1]:
        input_idx = torch.tensor([vocab.index(char)])
        _, hidden = model(input_idx, hidden)

    # 2. The Generation Loop
    curr_char = start_str[-1]
    for _ in range(length):
        input_idx       = torch.tensor([vocab.index(curr_char)])
        logits, hidden  = model(input_idx, hidden)

        # 3. Apply Temperature — higher temp = more random, lower temp = more confident
        probs    = F.softmax(logits / temperature, dim=-1)

        # 4. Multinomial Sampling: sample based on probability, not just argmax
        next_idx  = torch.multinomial(probs, num_samples=1).item()
        curr_char = vocab[next_idx]
        chars.append(curr_char)

    return "".join(chars)


if __name__ == "__main__":
    # "Sampling 20 characters starting with 's' at a creative temperature."
    model          = CharRNN(vocab_size=len(vocab), embed_size=32, hidden_size=128)
    generated_text = sample_from_model(model, vocab, start_str="s", length=20, temperature=0.7)

    print(f"Generated Sequence: {generated_text}")
    print("-" * 30)
    print("Successfully implemented a temperature-controlled inference loop!")
