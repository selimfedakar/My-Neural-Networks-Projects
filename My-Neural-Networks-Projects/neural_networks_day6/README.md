# 🧠 Broadcasting Rules & Vectorized Evaluation

I have moved beyond simple counting and entered the world of **Performance Optimization**. Today, I implemented the "Counting Solution" using **Broadcasting**, a powerful PyTorch feature that eliminates the need for slow Python `for` loops.

## 📸 My Notes
My notes today highlight the broadcasting rules and the transition point between modeling and evaluation.

![Broadcasting Logic](notes/page1.png)  

## 🚀 The Magic of Broadcasting
I documented the internal mechanics of how PyTorch handles tensors of different shapes:
- **Dimension Matching:** By using `keepdim=True`, I force the row sums into a $(27, 1)$ column vector.
- **Parallel Execution:** When I divide my $27 \times 27$ matrix by this column vector, PyTorch "virtually" expands the vector to perform the division across all elements simultaneously.
- **C-Level Speed:** This process bypasses the Python interpreter's overhead, running directly in highly optimized C code.

## 📉 Evaluation & Loss Foundations
I am transitioning from **building the model** to **evaluating its quality**.
- **Likelihood:** The probability assigned by the model to the actual real-world data.
- **Goal:** We want the probability of correct tokens to be very high, which mathematically corresponds to minimizing the **Loss**.

