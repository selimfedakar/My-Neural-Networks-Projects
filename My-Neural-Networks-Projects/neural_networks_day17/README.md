# Top-K Sampling - Filtering the Long Tail

I have further refined my inference engine by implementing Top-K Sampling. Today, I focused on "filtering the noise" to ensure my model stays within the realm of logical possibilities while maintaining its creative flair.


## The "Long Tail" Problem
I realized that even with a low temperature, the model might still assign a tiny probability to a character that makes no sense (the "long tail").
- **The Solution:** Instead of sampling from the entire vocabulary, I only look at the **K** most likely next characters.
- **The Filter:** I documented that we effectively set the probability of all other characters to zero (or -infinity in logits space) before the final Softmax.

##  Impact on Generation
- **K=1:** Equivalent to "Greedy Search" (always picks the best). The model becomes very repetitive.
- **K=5 to 10:** The "Gold Standard" for balance. The model stays creative but never picks a character that is completely out of context.
