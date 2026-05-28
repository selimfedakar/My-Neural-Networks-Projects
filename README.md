# My Neural Networks Projects

<p align="center">
  <img src="My-Neural-Networks-Projects/notes/page1.jpg" width="540" alt="Handwritten notes — Day 1" />
</p>

<p align="center"><i>Every concept handwritten first. Then implemented.</i></p>

---

A one-year, day-by-day study log of neural network internals — from a scalar autograd engine to character-level RNNs.

The work is documented through handwritten notes. Code supports the notes, not the other way around.

---

## What it is

This repository is a running implementation log. Each day covers one concept from the mathematical foundations of deep learning, explained in handwritten notes and backed by working Python code.

The path follows the architecture of modern language models: how a gradient flows, how a loss is minimized, how a model learns to generate text.

The main reference is Andrej Karpathy's lecture series on neural networks and language models.

---

## Why handwritten notes

In an era of AI-generated content, the handwritten notes are the signal. They show the actual reasoning process — where understanding broke down, where it clicked, what needed to be drawn to become clear.

The notes are not summaries of what I read. They are the place where I worked through the problem.

---

## What I worked on

I started from the very bottom — building a scalar autograd engine by hand, implementing the chain rule myself before ever touching PyTorch. From there I built a full neuron, layer, and MLP structure from scratch, wiring up the training loop manually so I understood exactly what `.backward()` is actually doing under the hood.

Once the foundations were solid, I moved into language modeling. I implemented a Bigram character model, then replaced it with an MLP using a sliding context window and an embedding lookup table. Along the way I worked through the details that most tutorials skip: why one-hot encoding biases the model, how the Log-Sum-Exp trick prevents softmax from exploding, how to split data properly and diagnose underfitting before touching hyperparameters.

The deeper I went, the more I focused on what breaks models internally. I went through weight initialization failures — the dead tanh problem, why a poorly scaled init can flatline a network before training even starts — then derived Kaiming initialization from first principles. I implemented Batch Normalization and Residual connections not as black-box layers but by understanding why they were invented and what problem each one solves.

I also built a diagnostic dashboard to visualize activation distributions and gradient flow across layers, which changed how I think about debugging neural networks entirely.

The current focus is Recurrent Networks — I have a working character-level RNN that generates text, and I am working through the sequence modeling fundamentals before moving to Transformers.

---

## Stack

| Layer | Technology |
|-------|-----------|
| Language | Python 3 |
| Framework | PyTorch |
| Environment | Jupyter / plain Python scripts |
| Dataset | Karpathy's `names.txt` (character-level) |

---

## Repo structure

```
My-Neural-Networks-Projects/
├── neural_networks_general/   autograd foundation (micrograd-style)
├── neural_networks_day2/      neural network core
├── neural_networks_day3/      training loop
├── ...
├── neural_networks_day17/     character-level RNN
├── neural_networks_day19/     extended experiments
└── notes/                     root handwritten notes (Day 1)
```

Each day folder contains:
```
neural_networks_dayN/
├── README.md     handwritten notes (images) + concept breakdown
├── notes/        scanned handwritten pages
└── *.py          supporting implementation
```

---

## Status

| Phase | Coverage | Status |
|-------|----------|--------|
| Autograd & backprop | Days General–4 | complete |
| Language modeling foundations | Days 5–9 | complete |
| Training best practices | Days 10–13 | complete |
| Modern architectures (BatchNorm, ResNet) | Days 14–15 | complete |
| Diagnostics & debugging | Day 16 | complete |
| Recurrent networks | Days 17–19 | in progress |
| Transformer architecture | — | planned |

---

**Ahmet Selim Fedakar** · Software Engineering · Los Angeles
