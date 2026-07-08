# --- START OF MANUAL BACKPROP ---

# 1. dlogprobs: Derivative of the loss w.r.t. the log-probabilities
# loss = -logprobs[range(n), Yb].mean()
# The derivative of a mean of N elements is 1/N. With the negative sign, it's -1/N.
# Only the indices of the correct characters (Yb) affected the loss, so others are 0.
dlogprobs = torch.zeros_like(logprobs)
dlogprobs[range(n), Yb] = -1.0 / n

# 2. dprobs: Backprop through the log() function
# logprobs = probs.log() -> d/dx log(x) = 1/x
dprobs = (1.0 / probs) * dlogprobs

# 3. dcounts & dcounts_sum_inv: Backprop through the division (multiplication by inverse)
# probs = counts * counts_sum_inv
dcounts = counts_sum_inv * dprobs # Partial derivative w.r.t. counts
dcounts_sum_inv = (counts * dprobs).sum(1, keepdim=True) # Summing because counts_sum_inv was broadcasted

# 4. dcounts_sum: Backprop through the reciprocal (1/x)
# counts_sum_inv = counts_sum**-1 -> d/dx (x^-1) = -x^-2
dcounts_sum = (-counts_sum**-2) * dcounts_sum_inv

# 5. dcounts (Branch 2): Backprop through the row summation
# counts_sum = counts.sum(1, keepdim=True) -> derivative of a sum is 1 (gradient flows to all inputs)
dcounts += torch.ones_like(counts) * dcounts_sum

# 6. dnorm_logits: Backprop through the exponential function
# counts = norm_logits.exp() -> d/dx exp(x) = exp(x)
dnorm_logits = counts * dcounts # We reuse 'counts' because counts = exp(norm_logits)

# 7. dlogits & dlogit_maxes: Backprop through the subtraction (norm_logits = logits - logit_maxes)
dlogits = dnorm_logits.clone() # Branch 1: direct flow from subtraction
dlogit_maxes = (-dnorm_logits).sum(1, keepdim=True) # Branch 2: summed because of broadcasting

# 8. dlogits (Branch 2): Backprop through the max() function (Routing)
# Only the element that was the maximum in each row gets the gradient.
dlogits += F.one_hot(logits.max(1).indices, num_classes=27) * dlogit_maxes

# 9. dh, dW2, db2: Linear Layer 2 backprop (Matrix multiplication and Bias)
# logits = h @ W2 + b2
dh = dlogits @ W2.T # Shape matching: (32,27) @ (27,64) -> (32,64)
dW2 = h.T @ dlogits # Shape matching: (64,32) @ (32,27) -> (64,27)
db2 = dlogits.sum(0) # Derivative of bias is the sum of gradients across the batch

# 10. dhpreact: Backprop through Tanh
# h = tanh(hpreact) -> d/dx tanh(x) = 1 - tanh(x)^2
dhpreact = (1.0 - h**2) * dh

# 11. dbngain, dbnbias, dbnraw: BatchNorm Scale & Shift backprop
# hpreact = bngain * bnraw + bnbias
dbngain = (bnraw * dhpreact).sum(0, keepdim=True)
dbnbias = dhpreact.sum(0, keepdim=True)
dbnraw = bngain * dhpreact

# 12. dbndiff, dbnvar_inv: BatchNorm Normalization step backprop
# bnraw = bndiff * bnvar_inv
dbndiff = bnvar_inv * dbnraw # Branch 1
dbnvar_inv = (bndiff * dbnraw).sum(0, keepdim=True)

# 13. dbnvar: Backprop through the sqrt and inverse (1 / sqrt(bnvar + eps))
# bnvar_inv = (bnvar + eps)**-0.5 -> d/dx x^-0.5 = -0.5 * x^-1.5
dbnvar = (-0.5 * (bnvar + 1e-5)**-1.5) * dbnvar_inv

# 14. dbndiff2: Backprop through the mean variance calculation
# bnvar = 1/(n-1) * bndiff2.sum(0, keepdim=True)
dbndiff2 = (1.0 / (n - 1)) * torch.ones_like(bndiff2) * dbnvar

# 15. dbndiff (Branch 2): Backprop through the square (x^2)
# bndiff2 = bndiff**2 -> d/dx x^2 = 2x
dbndiff += (2.0 * bndiff) * dbndiff2

# 16. dhprebn, dbnmeani: Backprop throughcentering (bndiff = hprebn - bnmeani)
dhprebn = dbndiff.clone() # Branch 1
dbnmeani = (-dbndiff).sum(0, keepdim=True)

# 17. dhprebn (Branch 2): Backprop through mean calculation
# bnmeani = 1/n * hprebn.sum(0, keepdim=True)
dhprebn += (1.0 / n) * torch.ones_like(hprebn) * dbnmeani

# 18. dembcat, dW1, db1: Linear Layer 1 backprop
# hprebn = embcat @ W1 + b1
dembcat = dhprebn @ W1.T
dW1 = embcat.T @ dhprebn
db1 = dhprebn.sum(0)

# 19. demb: Backprop through the view/reshape
# embcat = emb.view(32, 30)
demb = dembcat.view(emb.shape)

# 20. dC: Backprop through the embedding lookup (Index Assignment)
# Gradients add up for every time a specific character index was used.
dC = torch.zeros_like(C)
for k in range(Xb.shape[0]):
    for j in range(Xb.shape[1]):
        ix = Xb[k, j]
        dC[ix] += demb[k, j]

# --- END OF MANUAL BACKPROP ---