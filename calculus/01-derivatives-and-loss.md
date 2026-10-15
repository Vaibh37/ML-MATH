# Derivatives, the Chain Rule and Loss — First ML Calculus Lesson

**Studied and practiced on 15 October 2026.** Scope: derivative notation, basic rules, nested powers, simple partial derivatives, and building a squared-error loss from a prediction. Not yet trained a model.

## 1. Why derivatives matter

A derivative describes how a quantity changes when we vary an input. For neural network training, we care about how **loss** changes when a weight or bias changes. On a typical bowl-shaped loss curve, a negative derivative suggests increasing the chosen parameter locally moves downhill; a positive derivative suggests decreasing it moves downhill. The step size still matters.

Our earlier linear-regression demonstration used predictions $\hat y=wx+b$ for $x=[1,2,3,4]$, targets $y=[20,40,60,80]$, and fixed bias $b=0$. The resulting mean squared error is $L(w)=7.5(w-20)^2$. It reaches its minimum at $w=20$. The derivative is $dL/dw=15(w-20)$; we have discussed the direction of a potential update but have **not coded gradient descent**.

## 2. Read the notation before calculating

- $\frac{d}{dw}[3w+2]$ means differentiate the expression with respect to $w$.
- $\frac{dy}{dw}$ means how $y$ changes as $w$ changes when $y$ depends on $w$.
- $\frac{\partial L}{\partial b}$ means how loss changes as bias $b$ changes while other independent parameters are fixed.
- $L(w,b)$ means “the function named L accepts w and b”; parentheses are not multiplication.

There is no RHS-to-LHS rearrangement. Differentiation is a new operation on a function.

## 3. Rules with examples

**Constant and linear terms:** $d(7)/dw=0$ and $d(3w)/dw=3$.

**Power and sum rules:**

$$
\begin{aligned}
y&=4w^2+3w+7\\
\frac{dy}{dw}&=8w+3
\end{aligned}
$$

This exercise was solved independently.

## 4. Chain rule for nested expressions

When $y=[g(w)]^n$, first differentiate the outer power and then multiply by the derivative of the inner expression:

$$
\frac{dy}{dw}=n[g(w)]^{n-1}g'(w)
$$

Example: $y=(2w+5)^2$

$$
\begin{aligned}
\text{Outside: }&2(2w+5)\\
\text{Inside: }&2\\
\text{Result: }&4(2w+5)
\end{aligned}
$$

Example with a squared term inside a cube: $y=(2w^2+3w+1)^3$

$$
\begin{aligned}
\text{Outside: }&3(2w^2+3w+1)^2\\
\text{Inside: }&4w+3\\
\text{Result: }&3(2w^2+3w+1)^2(4w+3)
\end{aligned}
$$

Example: $y=(4w^2+3)^3$ has derivative $24w(4w^2+3)^2$. A practice attempt omitted the exponent on the outer parentheses; the corrected rule keeps it.

## 5. Partial derivatives: one input at a time

Given $f(w,b)=4w^2+5b+7$:

- $\partial f/\partial w = 8w$, since terms containing only $b$ are constant when $w$ changes.
- $\partial f/\partial b = 5$, since terms containing only $w$ are constant when $b$ changes.

Both results were practiced correctly.

## 6. Why write L(w,b)?

Start from the **prediction** for a single feature:

$$
\hat y=wx+b
$$

If the labeled training example is $x=4$ and $y=10$:

$$
\begin{aligned}
\hat y&=4w+b\\
L&=(\hat y-y)^2\\
L(w,b)&=(4w+b-10)^2
\end{aligned}
$$

The number 4 comes from input $x$, and 10 comes from target $y$. The variables $w$ and $b$ stay symbolic because we want to learn their values. This is the **squared-error loss for one example**, not the full-dataset MSE.

Use chain rule to differentiate:

$$
\begin{aligned}
\frac{\partial L}{\partial w}&=2(4w+b-10)\cdot4\\
\frac{\partial L}{\partial b}&=2(4w+b-10)\cdot1
\end{aligned}
$$

The separate derivatives tell us how sensitive the loss is to each parameter. Later, gradient descent will use them to make updates.

## 7. What was learned vs what is next

**Practiced:** basic power/sum rules, nested-expression chain rule (with a corrected exponent mistake), reading $dy/dx$ and $∂L/∂w$, partial derivative of a simple polynomial, and constructing $L(w,b)$ from a given $x$ and $y$.

**Still next:** reinforce gradients of MSE across several examples, implement updates of both $w$ and $b$, then derive sigmoid/ReLU derivatives and backpropagation for a 2→3→1 XOR network.

Related repositories: [Neural-Network](https://github.com/Vaibh37/Neural-Network) for network loss and [Python-cheat_sheet](https://github.com/Vaibh37/Python-cheat_sheet) for runnable NumPy syntax.
