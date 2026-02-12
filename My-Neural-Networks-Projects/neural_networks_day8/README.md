# 🧠 Softmax & The Log-Sum-Exp Trick for Stability

Today, I reached the point where I stop doing "manual math" and start using production-level loss functions. My focus was the **Softmax** function and the critical importance of **Numerical Stability**.

## 📸 Study Notes
![Softmax & Stability](notes/page1.png)
![Cross Entropy Logic](notes/page2.png)

## 🚀 Why Softmax?
To make sense of the network's raw outputs (Logits), I need valid probabilities. 
- **The Equation:** $P_i = \frac{e^{score_i}}{\sum e^{score_j}}$.
- **The Result:** This ensures all predictions are positive and sum exactly to $1.0$, creating a perfect probability distribution.

## ⚡ The "Infinity" Risk & F.cross_entropy
I documented why we don't calculate Softmax and Log-Likelihood separately in practice:
- **Memory Crashes:** If a network outputs a large score (like $100$), $e^{100}$ becomes so massive that the computer's memory fails.
- **The Solution:** I now use `F.cross_entropy`. This function uses a mathematical trick (Log-Sum-Exp) to calculate the same result without ever creating those massive, memory-breaking numbers.

## 📉 Selection, Not Distance
Unlike regression, which measures distance, I am focused on **Selection**. I care about the model picking the correct discrete category (character), which is why **Negative Log Likelihood** is my "Gold Standard" evaluation metric.
