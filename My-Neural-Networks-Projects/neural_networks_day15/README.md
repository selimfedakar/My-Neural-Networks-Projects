# Residual Networks - Breaking the Vanishing Gradient

I have integrated one of the most famous architectures in deep learning: ResNet. Today, I focused on how to make a signal travel through a massive "pipeline" without getting lost in the noise or shrinking to zero.

## 📸 My Notes
![Residual Connections & ResNet](notes/page1.png)

## 🌉 The Skip Connection (Residual)
I documented a modern innovation for the **Vanishing Gradient** problem. 
- **The Concept:** In a standard network, if a layer starts "killing" the gradient, every layer before it is also affected. A residual connection allows the signal to simply skip over the problematic part.
- **The Formula:** Instead of calculating $y = f(x)$, I implemented $y = x + f(x)$. This ensures that even if $f(x)$ becomes zero, the original signal $x$ still survives.
- **The Result:** The signal can still travel through other parts of the network without getting lost (skip). This allows me to stack hundreds of layers without the training becoming flat.

## 🏗️ The Modern Stack Consensus
I am now following the standard "pipeline" architecture for each block:
1. **Weight Layer** (Linear/Conv)
2. **Normalization** (Batch Norm)
3. **Non-Linearity** (ReLU/Tanh)

