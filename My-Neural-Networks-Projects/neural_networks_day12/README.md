# Day 12: Manual Backpropagation & The Courier Analogy

## 📸 My Notes
![Manual Backprop & Tanh](notes/page12.jpg)

I have moved beyond simply calling `.backward()` and started deconstructing the mathematical journey of a gradient. I now view backpropagation not as a black box, but as a precise delivery system where every neuron must pass a message to the next.


## 🚛 The Courier Journey (Chain Rule)
I documented that backpropagation is like a **courier** carrying a message from the loss function all the way back to the initial weights ($W_1$).
* **The Chain Rule:** I realized that the gradient of any variable is simply the **local gradient** multiplied by the **gradient from above**.
    $$\frac{\partial Loss}{\partial x} = \frac{\partial Loss}{\partial y} \cdot \frac{\partial y}{\partial x}$$
* **The Mission:** My goal is to ensure this "message" flows smoothly without becoming zero. If the courier's message (gradient) is lost or destroyed at any point, the weights at the start of the network will never learn.

## ⚠️ The Tanh Bottleneck & "Dead Tanh"
A major risk I identified is the behavior of the **Tanh** activation function during this journey.
* **The Local Gradient:** I documented the local gradient of tanh as $1 - \tanh^2(z)$.
* **The Death Zone:** I found that if my input $z$ is too large (e.g., $z = 20$), $\tanh(z)$ becomes almost $1.0$, which makes the derivative $0$.
* **The Result:** Mathematically, $0 \times (\text{gradient from above}) = 0$. The courier's message is destroyed at the tanh layer, the gradient becomes zero, and the neuron is effectively **"dead"**.

## 🧪 Verification: Manual vs. PyTorch
I implemented a manual calculation of these "courier messages" and verified them against PyTorch's automatic engine. By calculating $\frac{\partial Loss}{\partial w}$ as $x \cdot (1 - \tanh^2(z))$, I confirmed that my manual logic is identical to the production-grade `backward()` call.

---
*Next Step: Calibrating the signal volume with Kaiming Initialization.*