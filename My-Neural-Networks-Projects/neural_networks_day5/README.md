# 🧠Bigram Language Modeling & Probabilistic Sampling

I have officially moved from general neural network structures to **Natural Language Processing (NLP)**. Today, I implemented a Bigram model that learns the statistical frequency of character sequences to generate new text.

## 📸 My Notes
My notes today cover the transition to character-level modeling and the "pay off" of seeing a model generate its first tokens.

![Bigram Theory](notes/page1.png)
![Sampling Implementation](notes/page2.png)

## 🚀 Key Conceptual Breakthroughs
- **From Scalars to Batches:** I realized that while my scalar engine works for small tasks, production-scale AI requires **Batches** (random subsets of data) to handle millions of examples efficiently.
- **Bigram Counting:** I learned that the simplest way to "train" a language model is to count how often one character follows another. This frequency matrix becomes the model's "knowledge".
- **The Sampling Loop:** Using `torch.multinomial`, I created a loop that acts like a "weighted 27-sided die," picking the next character based on learned probabilities.

## 🧠 Why +1.0 and -1.0? (A Recap)
I am keeping my targets aligned with the `tanh` activation function because its **asymptote** nature handles classification boundaries perfectly. This foundation allows me to understand the probability distributions I am now building in this Bigram model.


