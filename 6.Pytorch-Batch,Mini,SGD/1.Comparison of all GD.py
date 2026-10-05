"""
1. Batch Gradient Descent
   → whole dataset → 1 update

2. Stochastic Gradient Descent (SGD)
   → 1 sample → 1 update

3. Mini-Batch Gradient Descent
   → small batch → 1 update


Batch:

Example 1 ─┐
Example 2 ─┤
Example 3 ─┤ → calculate gradient → update weight
Example 4 ─┤
Example 5 ─┘

SGD:

Sample 1 → gradient → update w
Sample 2 → gradient → update w
Sample 3 → gradient → update w
Sample 4 → gradient → update w
Sample 5 → gradient → update w

Mini-Batch:

Suppose you have 10 training examples and choose:

batch size = 3

The dataset is divided like this:

Mini-batch 1 → examples 1, 2, 3
Mini-batch 2 → examples 4, 5, 6
Mini-batch 3 → examples 7, 8, 9
Mini-batch 4 → example 10

Now the process is:

[1,2,3] → calculate gradient → UPDATE weights

[4,5,6] → calculate gradient → UPDATE weights

[7,8,9] → calculate gradient → UPDATE weights

[10] → calculate gradient → UPDATE weights

So for one epoch:

10 samples
↓
4 mini-batches
↓
4 weight updates


Batch GD
→ accurate/stable gradient
→ but can be slow and memory-heavy for huge datasets.

SGD
→ very frequent updates
→ but gradients are noisy.

SGD converges faster in terms of epochs because it makes many updates per epoch.
Batch GD is smoother but much slower when epochs are small.


Mini-Batch
→ faster computation
→ uses less memory
→ less noisy than SGD
→ works very well with GPUs and neural networks.

*An epoch means the model has gone through the entire training dataset once.


For modern machine learning and especially deep learning, Mini-Batch Gradient Descent is generally the practical choice.

Why:

Method:	                                    Practical use:

Batch GD	                Good for small datasets, but can be slow for large datasets

SGD	                        Simple and can learn quickly, but updates are noisy

Mini-Batch GD	            Best balance of speed, memory use, and stable learning



Problems and Advantages:-

# 1. BATCH GRADIENT DESCENT
# Problems:
# - Slow for large datasets
# - Requires more memory
# - Weight updates happen less frequently
#
# Advantage:
# - Stable and smooth gradient updates


# 2. STOCHASTIC GRADIENT DESCENT (SGD)
# Problems:
# - Very noisy gradient updates
# - Loss can go up and down
# - Does not converge smoothly
#
# Advantage:
# - Very frequent weight updates
# - Requires less memory


# 3. MINI-BATCH GRADIENT DESCENT
# Problems:
# - We must choose a suitable batch size
# - Small batch -> more noise
# - Large batch -> more memory usage

# Advantage:
# - Faster than Batch Gradient Descent
# - Less noisy than SGD
# - Uses GPU efficiently

# Deep Learning usually uses MINI-BATCH Gradient Descent.


"""