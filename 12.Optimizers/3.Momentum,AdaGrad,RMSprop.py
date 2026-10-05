"""
# Optimizers in Deep Learning
# SGD with Momentum, AdaGrad, RMSProp


==================================================
1. SGD (Stochastic Gradient Descent)
==================================================

Basic formula:

w = w - learning_rate × gradient

SGD uses the current gradient to update the weights.

Workflow:

Current gradient
      ↓
Calculate update
      ↓
Update weights


Problem with basic SGD:

- It can move slowly toward the minimum.
- It can zig-zag/oscillate in some directions.
- It only considers the current gradient.


==================================================
2. SGD with Momentum
==================================================

Momentum gives the optimizer a "memory" of previous updates.

Formula:

v_t = βv_(t-1) + g_t

w_t = w_(t-1) - ηv_t

Where:

v = velocity
β = momentum coefficient
g = current gradient
η = learning rate


Main idea:

Previous direction + Current gradient
              ↓
          Better update


Problem it solves:

- Reduces zig-zagging.
- Helps move faster in a consistent direction.
- Can reach the minimum faster than basic SGD.


Simple intuition:

Imagine a ball rolling downhill.

Without momentum:
The ball reacts mostly to the current slope.

With momentum:
The ball also remembers its previous direction,
so it gains speed in a consistent direction.


==================================================
3. AdaGrad
==================================================

AdaGrad gives different effective learning rates
to different parameters.

It keeps track of accumulated squared gradients.

Formula:

G_t = G_(t-1) + g_t²

Weight update:

w_t = w_(t-1) - [η / sqrt(G_t + ε)] × g_t


Main idea:

Large accumulated gradients
        ↓
Smaller effective learning rate

Small accumulated gradients
        ↓
Larger effective learning rate


Problem it solves:

Different parameters may need different learning rates.

For example:

Parameter A → receives large gradients
             → learning rate decreases more

Parameter B → receives small gradients
             → learning rate decreases less


Main problem with AdaGrad:

It keeps accumulating squared gradients forever.

G = g₁² + g₂² + g₃² + g₄² + ...

Therefore:

Accumulated gradient keeps increasing
              ↓
Effective learning rate keeps decreasing
              ↓
Learning can become extremely slow


==================================================
4. RMSProp
==================================================

RMSProp improves the main problem of AdaGrad.

Instead of accumulating ALL past squared gradients,
RMSProp uses an exponentially weighted moving average.

Formula:

v_t = βv_(t-1) + (1 - β)g_t²

Weight update:

w_t = w_(t-1) - [η / sqrt(v_t + ε)] × g_t


Main idea:

Recent gradients → more important
Old gradients    → less important


Why?

RMSProp does not allow the accumulated squared gradient
to keep growing forever like AdaGrad.

Therefore:

Recent gradient information
        ↓
Adaptive learning rate
        ↓
More stable learning


==================================================
5. Main Difference
==================================================

SGD:

Uses current gradient.

        ↓

SGD + Momentum:

Uses current gradient + previous direction.

        ↓

AdaGrad:

Uses accumulated squared gradients
to adapt the learning rate for each parameter.

        ↓

RMSProp:

Uses a moving average of squared gradients
instead of accumulating them forever.


==================================================
6. Comparison
==================================================

SGD
- Uses current gradient.
- Simple.
- Can be slow.
- Can zig-zag.

SGD with Momentum
- Remembers previous updates.
- Reduces oscillations.
- Speeds up movement in consistent directions.

AdaGrad
- Adaptive learning rate.
- Different parameters can have different effective
  learning rates.
- Problem: learning rate can become too small.

RMSProp
- Adaptive learning rate.
- Uses recent squared gradients.
- Prevents AdaGrad's learning rate from shrinking
  too aggressively.


==================================================
7. Easy Memory Trick
==================================================

SGD
→ Current gradient

Momentum
→ Previous direction + current gradient

AdaGrad
→ Accumulate squared gradients

RMSProp
→ Recent squared gradients


==================================================
8. Interview Answer
==================================================

SGD updates weights using the current gradient.

SGD with Momentum adds memory of previous updates
to reduce oscillations and accelerate convergence.

AdaGrad adapts the learning rate for each parameter
using accumulated squared gradients, but its learning
rate can eventually become too small.

RMSProp improves AdaGrad by using an exponentially
weighted moving average of squared gradients, giving
more importance to recent gradients and reducing the
problem of the learning rate becoming extremely small.


Simple Analogy:
GD: Walking downlhill carefully, step by step.
SGD: Jumping randomly down the hill.
Momentum: Jumping down the hill with a backpack that remembers your previous jumps.
AdaGrad: Jumping down the hill, but slowing down if you keep jumping in the same direction.Smart Step Size
RMSProp: Forgetting old jumps and focusing on recent jumps, allowing for a more adaptive step size.
Adam:   Combining momentum and adaptive learning rates for efficient optimization.
        It means that Adam combines the benefits of both Momentum and RMSProp, 
        making it a popular choice for many deep learning tasks.
        Smart+Fast+Stable
"""