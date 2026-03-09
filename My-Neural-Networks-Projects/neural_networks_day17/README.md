# Character-Level RNN - Building a Text Generator
from my perspective, lets get into it. 
Today, I transitioned from a single RNN cell to a complete Language Model. I am now training a network to predict the next character in a sequence, effectively teaching it the "structure" of language from scratch.

## 🔤 The Character-to-Index Pipeline
I realized that neural networks cannot read letters; they only understand tensors.
- **Vocabulary:** I built a unique mapping of every character in my dataset to an integer index.
- **One-Hot Encoding vs. Embeddings:** While I used one-hot earlier, I am now moving toward a more efficient 'lookup' approach to represent my characters in a high-dimensional space.

## 🔄 Temporal Memory (The Unrolling)
Unlike standard networks, the RNN processes a "stream" of data.
- **Hidden State ($h_t$):** This is the model's memory. I documented that $h_t$ is updated at every step using the current character and the memory of all previous characters.
- **The Equation:** I am implementing the recurrence: 
  $$h_t = \tanh(W_{ih}x_t + W_{hh}h_{t-1} + b)$$
- **Sequence Bottleneck:** I identified that the longer the sequence, the harder it is for the "courier" (gradient) to travel back to the first character without vanishing.

## 🎯 The Goal: Sampling
Success for me today is not just a low loss, but the ability to "sample" from the model—asking it to generate a word it has never seen before based on the probability distribution it learned.