# Day 13: Kaiming Initialization - Calibrating the Signal Volume

## 📸 My Notes
![Kaiming Initialization & Fan-in](notes/page1.png)


I have progressed from using the "0.01 sweet spot" to implementing a mathematically sound initialization strategy. I am now focused on keeping the "volume" of my network's signal consistent as it travels through deep layers, ensuring the "soul" of the neural network—the non-linearity—can actually work.

## The Square Root of Fan-in Rule
I realized that if I initialize a layer with a standard Gaussian distribution ($\sigma=1$), the output variance will grow proportionally to the number of inputs.
- **Fan-in:** This represents the number of input connections coming into a single neuron.
- **The Solution:** I am simplifying the "black magic" of math into a professional rule of thumb: $W_{scaled} = \frac{W_{random}}{\sqrt{fan\_in}}$.
- **Result:** This ensures the output activations remain roughly Gaussian (unit variance) throughout the network, preventing the signal from growing exponentially or shrinking to zero.

## Avoiding the Vanishing Gradient
I documented that correct initialization is my primary defense against the **Vanishing Gradient** ($kaybolan$) problem. 
- **The Goal:** I want the signal to reach all the way back to the first layer ($W_1$) without becoming zero. 
- **Active Tanh:** By keeping the activation variance stable, I ensure that inputs to the `tanh` layer stay in the active region rather than the "dead zones" where the gradient is 0.

