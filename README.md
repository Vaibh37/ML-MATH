# Mathematics for Machine Learning

Practical mathematics notes for computer science, machine learning, deep learning, and ML engineering.

The focus is on **what each concept means in code**, **how to represent it with NumPy**, and **where it appears in ML systems**. These are learning notes, not a full university mathematics syllabus.

## Study structure

### Linear algebra

1. [Vectors — Introduction (CS perspective)](linear-algebra/01-vectors-introduction.md)
2. [Vectors — Worked lessons and formula cheatsheet](linear-algebra/vectors/README.md) — addition, subtraction, scalar multiplication, linear combinations, dot product, magnitude, normalization. Studied 14 October 2026.
3. [Angles and cosine similarity](linear-algebra/vectors/07-angles-and-cosine-similarity.md) — reviewed and practiced.
4. [Matrices and broadcasting](linear-algebra/matrices/README.md) — shapes, arithmetic, multiplication, transpose, and bias addition.
5. [Trigonometry tables](trigonometry/TRIG_CHEATSHEET.md) — normal and inverse functions with a PNG reference.

Projection, eigenvectors, and deeper statistics remain future topics. The introductory calculus below has now been practiced.

### Calculus (updated 18 October 2026)

- [Derivative and loss quick cheatsheet](calculus/00-derivatives-cheatsheet.md) — notation, power rule, chain rule and partial derivatives.
- [Derivatives applied to squared-error loss](calculus/01-derivatives-and-loss.md) — worked examples, including where `L(w,b)` comes from.
- [Gradient descent from a one-example loss](calculus/02-gradient-descent-single-sample.md) — derivation of gradients, five checked updates, learning rate and overshooting.
- [Dataset MSE gradients and ReLU chain rule](calculus/03-batch-mse-and-relu-chain-rule.md) — full-batch gradient derivation, vectorization, and the first gradient through activation.
- [Full XOR chain-rule derivation](calculus/04-xor-backpropagation-chain-rule.md) — output and hidden-layer derivatives, gradient shapes, and all 13 parameter updates.
- **Practiced and coded (16 October):** single-example and full-batch linear-neuron training. **Completed (18 October):** derived and implemented full two-layer XOR backpropagation, including six hidden weight gradients, three hidden bias gradients, three output weight gradients, and one output bias gradient. See [the NumPy implementation](https://github.com/Vaibh37/Neural-Network/blob/main/examples/xor_network.py). Numerical checking of all gradients remains deferred.

### Later

- Calculus next: numerical gradient checking (deferred), then Jacobians and vectorized derivative reasoning when needed.
- Probability and statistics: distributions, expectation, variance, conditional probability.

## Method

1. Understand the idea in plain language.
2. Represent it as data in Python/NumPy.
3. Predict the output before executing the code.
4. Relate the operation to an ML application.
5. Mark unfamiliar topics as **next**, rather than claiming they are complete.

NumPy syntax reference: [Python-cheat_sheet/NUMPY_CHEATSHEET.md](https://github.com/Vaibh37/Python-cheat_sheet/blob/main/NUMPY_CHEATSHEET.md).

**Goal:** Understand enough mathematics to reason about how ML models operate and to implement the underlying calculations.
