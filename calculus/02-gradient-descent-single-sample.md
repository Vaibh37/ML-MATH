# 02 — Gradients and Gradient Descent for One Training Example

**Practiced: 16 October 2026.** This lesson connects the chain rule to real Python training updates. It covers a **single-example linear model**, not a multi-layer neural network.

## 1. Prediction and loss are different numbers

A linear neuron without an activation function predicts:

~~~math
\hat y=wx+b
~~~

For one example, use x = 2, target y = 10, initial weight w = 3, and bias b = 1.

- Prediction: 3 × 2 + 1 = **7**.
- Error: 7 − 10 = **−3**.
- Squared-error loss: (7 − 10)² = **9**.

**The goal is prediction → 10 and loss → 0**, not loss → 10. If prediction becomes 8, squared loss becomes 4; if it becomes 9, squared loss becomes 1.

## 2. Why the weight gradient includes x

Substitute the prediction into the loss:

~~~math
L(w,b)=(wx+b-y)^2.
~~~

The labels x and y are fixed for this example. When differentiating with respect to w, hold x, y, and b constant.

Use the chain rule: derivative of the outer square × derivative of the inside.

~~~math
\begin{aligned}
\frac{\partial L}{\partial w}
&=2(wx+b-y)\frac{\partial}{\partial w}(wx+b-y)\\
&=2(wx+b-y)\cdot x\\
&=2(\hat y-y)x.
\end{aligned}
~~~

**Where x came from:** the derivative of wx with respect to w is x, just as the derivative of 3w is 3. The derivative of b − y is 0.

For bias, hold w, x, and y fixed. The derivative of the inside with respect to b is 1:

~~~math
\frac{\partial L}{\partial b}
=2(wx+b-y)\cdot1
=2(\hat y-y).
~~~

At w = 3, b = 1, x = 2, and y = 10:

- Weight gradient: dw = 2(7 − 10) × 2 = **−12**.
- Bias gradient: db = 2(7 − 10) = **−6**.

A negative gradient suggests increasing that parameter will locally reduce loss; it does **not** guarantee that an arbitrarily large update is safe.

## 3. Update the parameters

Choose the learning rate alpha = 0.05. It is a **hyperparameter** selected by us, not derived from the squared-error expression.

~~~math
\begin{aligned}
w_{\mathrm{new}}&=w-\alpha\frac{\partial L}{\partial w}
=3-0.05(-12)=3.6\\
b_{\mathrm{new}}&=b-\alpha\frac{\partial L}{\partial b}
=1-0.05(-6)=1.3.
\end{aligned}
~~~

The next prediction is 3.6 × 2 + 1.3 = **8.5**. The new loss is (8.5 − 10)² = **2.25**. This is one training update.

## 4. Repeat: five checked updates

Each row below uses the *updated* parameters and recomputes the loss.

| After update | w | b | Prediction | Squared loss |
| ---: | ---: | ---: | ---: | ---: |
| 0 (initial) | 3 | 1 | 7 | 9 |
| 1 | 3.6 | 1.3 | 8.5 | 2.25 |
| 2 | 3.9 | 1.45 | 9.25 | 0.5625 |
| 3 | 4.05 | 1.525 | 9.625 | 0.140625 |
| 4 | 4.125 | 1.5625 | 9.8125 | 0.03515625 |
| 5 | 4.1625 | 1.58125 | 9.90625 | 0.0087890625 |

The values are rounded for display. Python floats may contain small representation differences.

## 5. Why the learning rate matters

Consider a **different** practice loss: L(w,b) = (4w + 2b − 10)², initially w = 1, b = 2, and loss = 4. Its gradients are dw = −16, db = −8.

- With alpha = **0.1**, updated (w,b) = (2.6,2.8), and **loss rises to 36**: an overshoot.
- With alpha = **0.01**, updated (w,b) = (1.16,2.08), and **loss falls to 1.44**.

**Correct derivatives do not guarantee every step improves the loss.** Check the loss after updates.

## 6. Manual differentiation vs automatic differentiation

- **NumPy and plain Python:** derive gradient formulas by hand, then write the formulas as code. This is what the from-scratch project uses.
- **SymPy:** can derive symbolic expressions such as derivatives.
- **PyTorch autograd:** tracks tensor operations and computes gradients from them with backward(). Automatic differentiation is still calculus, but the framework evaluates the chain rule.

We are doing this manually *to understand what backpropagation must compute*. A single linear neuron is much simpler than a 2→3→1 XOR network.

## 7. Remember the full training cycle

**Predict → calculate loss → calculate gradients → update parameters → predict again.**

- Gradient calculation measures sensitivity; parameter updates are what change the model.
- **Epoch:** one pass over a dataset. With one example, one loop iteration is one epoch.
- A Python loop **does not reset w and b**. Only the initializer before the loop sets their original values; each assignment inside the loop replaces the previous value.

Next: [one-neuron training-loop notes](https://github.com/Vaibh37/Neural-Network/blob/main/notes/06-first-gradient-descent-training-loop.md) and [runnable code](https://github.com/Vaibh37/Neural-Network/blob/main/examples/train_single_neuron.py). **Not yet completed:** dataset-level MSE gradients and backpropagation through hidden layers.
