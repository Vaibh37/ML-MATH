# Derivatives for ML — Quick Cheat Sheet

**Checkpoint: 15 October 2026.** Basic differentiation and partial derivatives have been practiced. Gradient descent updates and backpropagation have **not** been implemented.

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

$
y=(4w^2+3)^3
dy/dw = 3(4w^2+3)^2 · 8w
      = 24w(4w^2+3)^2
$

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

$
prediction: ŷ = 4w+b
squared-error loss: L(w,b) = (ŷ - y)^2 = (4w+b-10)^2
$

The derivative of the one-example loss follows the chain rule:

$
∂L/∂w = 2(4w+b-10)·4 = 8(4w+b-10)
∂L/∂b = 2(4w+b-10)·1 = 2(4w+b-10)
$

This is a **single-sample squared-error example**, not yet a batch MSE gradient.

For interpretation and full worked lessons, see [01-derivatives-and-loss.md](01-derivatives-and-loss.md). Loss examples: [Neural-Network](https://github.com/Vaibh37/Neural-Network).
