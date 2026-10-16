# Derivatives for ML — Quick Cheat Sheet

**Checkpoint: 16 October 2026.** Single-example and four-example dataset gradients and parameter updates have been practiced. ReLU derivative and one chain-rule step introduced. Hidden-layer backpropagation has **not** been implemented.

## Reading the notation

| Notation | Read it as | Meaning |
| --- | --- | --- |
| $\frac{d}{dw}f(w)$ | Differentiate f with respect to w | Treat w as the changing variable |
| $\frac{dy}{dx}$ | Derivative of y with respect to x | How y changes as x changes |
| $\frac{\partial L}{\partial w}$ | Partial derivative of loss with respect to w | Vary w, hold other independent variables fixed |
| $L(w,b)$ | Loss as a function of w and b | Parentheses identify inputs, not multiplication |

**Do not move RHS terms to the LHS.** Differentiation is an operation, not rearranging an equation. $d/dx$ and $∂/∂w$ are notation for differentiation, not ordinary fractions to evaluate by dividing letters.

## Essential rules

Assume $c$ is constant and the derivative is with respect to $w$:

| Rule | Example |
| --- | --- |
| Constant: derivative 0 | $d(7)/dw = 0$ |
| Linear: derivative of w is 1 | $d(3w)/dw = 3$ |
| Power: $d(w^n)/dw = nw^{n-1}$ | $d(4w^2)/dw = 8w$ |
| Sum: differentiate each term | $d(4w^2+3w+7)/dw = 8w+3$ |
| Chain: outer derivative × inner derivative | $d[(2w+5)^2]/dw = 4(2w+5)$ |

Nested example:

```math
\begin{aligned}
y &= (4w^2+3)^3\\
\frac{dy}{dw} &= 3(4w^2+3)^2\cdot8w\\
&= 24w(4w^2+3)^2
\end{aligned}
```

**Common slip:** Keep the outside exponent $2$ after differentiating the cube; do not drop it.

## Partial derivatives

For $f(w,b)=4w^2+5b+7$:

- $∂f/∂w = 8w$ (the $b$ terms are constant).
- $∂f/∂b = 5$ (the $w$ terms are constant).

For $\hat y=wx+b$, with $x$, $w$ and $b$ treated as independent inputs:

- $∂\hat y/∂w = x$
- $∂\hat y/∂b = 1$
- $∂\hat y/∂x = w$

## From prediction to loss

Given one labeled example $x=4$, $y=10$ and prediction $\hat y=wx+b$:

```math
\begin{aligned}
\hat y &=4w+b\\
L(w,b)&=(\hat y-y)^2=(4w+b-10)^2
\end{aligned}
```

The derivative of the one-example loss follows the chain rule:

```math
\begin{aligned}
\frac{\partial L}{\partial w}&=2(4w+b-10)\cdot4=8(4w+b-10)\\
\frac{\partial L}{\partial b}&=2(4w+b-10)\cdot1=2(4w+b-10)
\end{aligned}
```

This is a **single-sample squared-error example**. The full-batch MSE extension is documented in [lesson 03](03-batch-mse-and-relu-chain-rule.md).

For interpretation and full worked lessons, see [01-derivatives-and-loss.md](01-derivatives-and-loss.md). Loss examples: [Neural-Network](https://github.com/Vaibh37/Neural-Network).


## Gradient descent from one-example squared error (16 October)

For prediction y-hat = wx + b and **one** target y:

~~~math
\begin{aligned}
L&=(\hat y-y)^2,\\
\frac{\partial L}{\partial w}&=2(\hat y-y)x,\\
\frac{\partial L}{\partial b}&=2(\hat y-y).
\end{aligned}
~~~

**Why x?** In d(wx)/dw, x is constant. **Why not x in the bias gradient?** In d(wx+b−y)/db, the inside derivative is 1.

Gradient descent: new weight = old weight − (learning rate × weight gradient). Same for bias. With x=2, target y=10, initial w=3, b=1, and learning rate 0.05: the first gradients are dw=−12, db=−6; updated w=3.6, b=1.3; prediction improves 7→8.5 and loss decreases 9→2.25.

**Watch out:** loss is the error measure, not the desired target. Large learning rates can increase loss. Detailed example: [02-gradient-descent-single-sample.md](02-gradient-descent-single-sample.md).


## Full-batch gradients (four examples; 16 October)

For n examples with predictions y-hat_i = wx_i + b and dataset MSE:

~~~math
L(w,b)=\frac1n\sum_{i=1}^n(\hat y_i-y_i)^2
~~~

~~~math
dw=\frac{\partial L}{\partial w}=\frac2n\sum_{i=1}^n(\hat y_i-y_i)x_i
~~~

~~~math
db=\frac{\partial L}{\partial b}=\frac2n\sum_{i=1}^n(\hat y_i-y_i)
~~~

With x=[1,2,3,4], y=[5,10,15,20], w=3, b=1: dw=−25, db=−8, MSE=21. At alpha=0.05, one correct update gives w=4.25, b=1.4 and MSE=0.92875. Five updates give MSE ≈ 0.325768.

## ReLU chain rule (introduced; not multi-layer backprop)

z=wx+b, a=ReLU(z), L=(a−y)^2.

~~~math
\frac{\partial L}{\partial z}
=\frac{\partial L}{\partial a}\cdot\frac{\partial a}{\partial z}
=2(a-y)\operatorname{ReLU}'(z).
~~~

ReLU'(z) = 0 for z<0; 1 for z>0; undefined at z=0 (often 0 in code). Read the detailed derivation in [lesson 03](03-batch-mse-and-relu-chain-rule.md). Next: gradients through w and b of an activated neuron, then hidden layers.
