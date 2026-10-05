"""
Important to remember

Regression: How far is the predicted numerical value from the actual value?

Classification: How far is the predicted probability or decision from the correct class?

Unsupervised Learning: How well does the model achieve its learning objective,
                       such as forming clusters, reconstructing data, or learning representations?

Also, a loss function is not always the same as an evaluation metric. For example,
K-Means minimizes WCSS, while Silhouette Score is an evaluation metric used to 
assess cluster separation and cohesion.

Loss function → tells the model/algorithm what to minimize or optimize during training.
Evaluation metric → tells us how well the final model or clustering result performed.

# LOSS FUNCTIONS

Loss Function:
→ Measures how much the model's prediction differs
from the actual/desired result.

==================================================

# 1. REGRESSION

==================================================

Regression predicts continuous numerical values.

Examples:
→ House price
→ Salary
→ Temperature
→ Student marks

## 1.1 MSE — Mean Squared Error

Formula:

MSE = (1/n) * Σ(y - ŷ)^2

y  = Actual value
ŷ  = Predicted value

How it works:

1. Calculate the error.
2. Square the error.
3. Take the average.

Example:

Actual = 100
Predicted = 90

Error = 100 - 90
= 10

Squared Error = 10²
= 100

Important:

→ Large errors are penalized more heavily.
→ Sensitive to outliers.
→ Commonly used in regression.
→ Differentiable
→ Easy to interpret
→ It has only single local minima

## 1.2 MAE — Mean Absolute Error

Formula:

MAE = (1/n) * Σ|y - ŷ|

How it works:

1. Calculate the error.
2. Take the absolute value.
3. Take the average.

Example:

Actual = 100
Predicted = 90

Error = |100 - 90|
= 10

Important:

→ Less sensitive to outliers than MSE.
→ Easy to understand.
→ Error remains in the same unit as the target.
→ Non-differentiable at zero

## 1.3 Huber Loss

Huber Loss combines the behavior of:

→ MSE for small errors
→ MAE for large errors

Main idea:

Small error
↓
MSE-like behavior

Large error
↓
MAE-like behavior

Important:

→ More robust to outliers than MSE.
→ Useful when the dataset contains outliers.
→ Mainly used for regression.
→Requires tuning an extra hyperparameter (delta)
→Slower computation

==================================================

# 2. CLASSIFICATION

==================================================

Classification predicts classes/categories.

Examples:

→ Pass / Fail
→ Spam / Not Spam
→ Cat / Dog
→ Disease / No Disease

## 2.1 BCE — Binary Cross-Entropy

Used for:

→ Binary Classification

Binary means:

2 classes

Example:

0 = Fail
1 = Pass

Formula:

BCE = -(1/n) * Σ[
y log(ŷ) +
(1-y) log(1-ŷ)
]

y  = Actual label
ŷ  = Predicted probability

Example:

Actual = 1
Predicted probability = 0.90

→ Model is confident about the correct class.
→ Low loss.

Actual = 1
Predicted probability = 0.10

→ Model is confident about the wrong class.
→ High loss.

Important:

→ BCE is commonly used for binary classification.
→ It works with predicted probabilities.

## 2.2 CCE — Categorical Cross-Entropy

Used for:

→ Multi-class Classification

Formula:
L(y,ŷ)= -Σ[ylog(ŷ)]


Example:

0 = Cat
1 = Dog
2 = Horse

Model prediction:

Cat   = 0.10
Dog   = 0.80
Horse = 0.10

If actual class = Dog:

→ High probability is given to the correct class.
→ Loss is low.

Usually used with:

→ Softmax output

Important:

→ Works naturally with probabilities
→ Highly effective for multiclass-classification
→ Sensitive for Mislabeled data
→ Class imbalance sensitive
→ Needs One-hot encoding



BCE
→ Binary classification

CCE
→ Multi-class classification


==================================================

# 3. UNSUPERVISED / GENERATIVE DEEP LEARNING

==================================================

GAN = Generative Adversarial Network

GANs can learn from unlabeled real data.

A GAN contains:

1. Generator
2. Discriminator

## 3.1 Generator

Generator creates fake data.

Random Noise
↓
Generator
↓
Fake Data

Example:

Random Noise
↓
Generator
↓
Fake Image

Goal:

→ Create fake data that looks real.

## 3.2 Discriminator

Discriminator receives:

→ Real data
→ Fake data

It predicts:

Real → 1
Fake → 0

Goal:

→ Distinguish real data from fake data.

==================================================

# 3.3 DISCRIMINATOR LOSS

==================================================

Formula:

L_D =
-[log(D(x)) + log(1 - D(G(z)))]

Where:

x = Real data

z = Random noise

G(z) = Fake data created by Generator

D(x) = Probability that real data is real

D(G(z)) = Probability that fake data is real

Discriminator wants:

D(real) → 1

D(fake) → 0

Example:

D(real) = 0.90
D(fake) = 0.10

→ Correct predictions
→ Low Discriminator Loss

If:

D(real) = 0.10
D(fake) = 0.90

→ Wrong predictions
→ High Discriminator Loss

==================================================

# 3.4 ORIGINAL MINIMAX GAN LOSS

==================================================

Original GAN objective:

min_G max_D V(D,G)

where:

V(D,G) =
E[log(D(x))]
+
E[log(1 - D(G(z)))]

Discriminator:

→ Tries to MAXIMIZE the objective.

Generator:

→ Tries to MINIMIZE the objective.

Discriminator wants:

D(real) → 1
D(fake) → 0

Generator wants:

D(fake) → 1

Simple idea:

Generator:
"I want to fool the Discriminator."

Discriminator:
"I want to detect the fake data."

==================================================

# FINAL SUMMARY

==================================================

## REGRESSION

1. MSE
   → Large errors are penalized heavily.

2. MAE
   → More robust to outliers than MSE.

3. Huber Loss
   → MSE-like for small errors,
   MAE-like for large errors.

## CLASSIFICATION

1. BCE
   → Binary classification.

2. CCE
   → Multi-class classification.

3. Huber Loss
   → Mainly a regression loss,
   NOT a standard classification loss.

## UNSUPERVISED / GENERATIVE

GAN:

1. Discriminator Loss
   → Discriminator learns:
   Real → 1
   Fake → 0

2. Original Minimax GAN Loss
   → Discriminator maximizes.
   → Generator minimizes.

Generator:
→ Create fake data that looks real.

Discriminator:
→ Distinguish real data from fake data.














"""