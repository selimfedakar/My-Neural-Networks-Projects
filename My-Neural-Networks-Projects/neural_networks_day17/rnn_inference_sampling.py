import torch
import torch.nn.functional as F


# My Goal: Generate a sequence of characters using 'Temperature' to control creativity.

def sample_from_model(model, start_str, length, temperature=1.0):
    model.eval()  # "I am now in evaluation mode"
    chars = [c for c in start_str]
    hidden = model.init_hidden()

    # 1. Warm up the hidden state with the starting string
    for char in start_str[:-1]:
        input_idx = torch.tensor([vocab.index(char)])
        _, hidden = model(input_idx, hidden)

    # 2. The Generation Loop
    curr_char = start_str[-1]
    for _ in range(length):
        input_idx = torch.tensor([vocab.index(curr_char)])
        logits, hidden = model(input_idx, hidden)

        # 3. Apply Temperature
        # "I divide logits by temperature to control the 'confidence' of the model."
        probs = F.softmax(logits / temperature, dim=-1)

        # 4. Multinomial Sampling: Instead of argmax, I sample based on probability
        next_idx = torch.multinomial(probs, num_samples=1).item()
        curr_char = vocab[next_idx]
        chars.append(curr_char)

    return "".join(chars)


# --- THE TEST RUN ---
# Assuming the CharRNN model and vocab from the previous session are loaded
# "I am sampling 20 characters starting with 's' at a creative temperature."
generated_text = sample_from_model(model, start_str="s", length=20, temperature=0.7)

print(f"Generated Sequence: {generated_text}")
print("-" * 30)
print("I've successfully implemented a temperature-controlled inference loop!")