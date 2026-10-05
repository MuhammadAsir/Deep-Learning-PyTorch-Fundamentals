"""

# EWMA (Exponentially Weighted Moving Average)


# EWMA is used to SMOOTH noisy/fluctuating values.
#
# It gives MORE importance to recent values
# and LESS importance to older values.



# WHY DO WE NEED EWMA?
# ------------------------------------------------------------

# Sometimes values fluctuate a lot because of NOISE.

# Example:

# Raw values:
# 10 -> 2 -> 8 -> 3 -> 9 -> 1 -> 7

# These values are jumping up and down.
#
# EWMA smooths these fluctuations and gives
# a more stable trend.

# FORMULA
# ------------------------------------------------------------

# EWMA = beta * previous_EWMA + (1 - beta) * current_value

# beta controls how much we remember the previous value.

# beta close to 1:
# -> More smoothing
# -> More importance to previous values

# beta close to 0:
# -> Less smoothing
# -> More importance to current value



# SIMPLE EXAMPLE
# ------------------------------------------------------------

# beta = 0.9
# previous_EWMA = 10
# current_value = 2

# EWMA = 0.9 * 10 + 0.1 * 2
#      = 9 + 0.2
#      = 9.2

# Instead of changing suddenly:

# 10 -> 2

# EWMA changes smoothly:

# 10 -> 9.2



# WHEN IS EWMA MOST NEEDED?
------------------------------------------------------------

# EWMA is MOST useful when:

# 1. Values are very noisy.
# 2. Values fluctuate a lot.
# 3. We want a smooth/stable trend.
# 4. We don't want to react too much to one sudden change.


# In Deep Learning:

# SGD
#   ↓
# Noisy gradients
#   ↓
# EWMA
#   ↓
# Smoother gradient information


# WHERE IS EWMA USED IN DEEP LEARNING?
------------------------------------------------------------

# 1. Momentum
#    Uses a moving average of past gradients.

# 2. Adam Optimizer
#    Uses exponential moving averages of:
#    - gradients
#    - squared gradients

# 3. Batch Normalization
#    Uses running averages of mean and variance
#    for inference.



# EASY MEMORY TRICK
============================================================

# EWMA = SMOOTHING NOISY VALUES

# More noise -> More useful EWMA

# Recent value gets MORE importance.
# Older values get LESS importance.
# ============================================================

Easy interview answer:
EWMA is most useful when data or gradients are noisy and fluctuate significantly. 
It smooths these fluctuations by giving more weight to recent values while retaining 
some information from previous values.


| Field                                  | How EWMA is used |
|---|---|
|📈Finance & Trading                     | Very common — smoothing stock prices, volatility, risk |
|📊Statistics / Time Series              | Very common — smoothing noisy time-series data |
|🤖Machine Learning / Deep Learning      | Common — Adam, Momentum, BatchNorm running statistics |
|🌐Networking                            | Common — estimating latency, traffic, packet rates |
|🏭Industrial monitoring                 | Common — smoothing sensor measurements |
"""