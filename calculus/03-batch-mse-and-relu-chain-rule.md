# 03 — Dataset MSE Gradients and the ReLU Chain Rule

**Checkpoint:** 16 October 2026.

**Practiced:** Differentiating dataset MSE with respect to one shared weight and bias; calculating full-batch gradients in NumPy; a first chain-rule step through ReLU. **Not yet completed:** gradients through all weights in a hidden layer.

## Notation and meanings

| Notation | Plain meaning |
| --- | --- |
| x_i | Input of example i |
| y_i | Target of example i |
| y-hat_i | Prediction for example i |
| n | Number of training examples (here, 4) |
| w, b | One neuron's shared weight and bias |
| L(w,b) | Mean squared error as a function of parameters |
| ∂L/∂w (dw) | Weight gradient: how L changes with w |
| ∂L/∂b (db) | Bias gradient: how L changes with b |
| alpha | Learning rate used to change the parameters |

The symbol ∑ means "sum the terms over training examples." A subscript i means one specific example; it does not mean a new neuron.

## From single-example squared error to dataset MSE

For each example: y-hat_i = w*x_i + b and error_i = y-hat_i - y_i.

For one example:

~~~math
L_i=(wx_i+b-y_i)^2
~~~

The chain rule with respect to w and b gives:

~~~math
\frac{\partial L_i}{\partial w}=2(wx_i+b-y_i)x_i,
\qquad
\frac{\partial L_i}{\partial b}=2(wx_i+b-y_i).
~~~

For **n** examples, mean squared error is the average of the n single-example losses:

~~~math
L(w,b)=\frac{1}{n}\sum_{i=1}^n(wx_i+b-y_i)^2.
~~~

Because differentiation distributes over a finite sum, and 1/n is constant, we can average the **individual gradients**:

~~~math
\frac{\partial L}{\partial w}
=\frac{2}{n}\sum_{i=1}^n(wx_i+b-y_i)x_i
~~~

~~~math
\frac{\partial L}{\partial b}
=\frac{2}{n}\sum_{i=1}^n(wx_i+b-y_i).
~~~

This is *not* calculating a mean prediction error and then squaring it; nor is it averaging the weights. It is averaging contributions to the **same weight's gradient** across data examples.

## Numerical worked example

Four examples: x = [1, 2, 3, 4], y = [5, 10, 15, 20], w = 3, b = 1.

Predictions: [4, 7, 10, 13].

Errors (prediction − target): [-1, -3, -5, -7].

Squared errors: [1, 9, 25, 49]. MSE = 21.

Per-example weight gradients are:

~~~math
2(\hat y_i-y_i)x_i=[-2,-12,-30,-56].
~~~

Average = dw = -25.

Bias gradients are:

~~~math
2(\hat y_i-y_i)=[-2,-6,-10,-14].
~~~

Average = db = -8.

At learning rate alpha = 0.05:

~~~math
w_{\rm new}=3-0.05(-25)=4.25
~~~

~~~math
b_{\rm new}=1-0.05(-8)=1.4
~~~

Both parameter updates are based on the old parameters' gradients. Recompute predictions with w = 4.25, b = 1.4: post-update MSE = **0.92875**.

Five epochs of full-batch gradient descent reach MSE ≈ **0.325768**. A lower loss is the goal, but a too-large step can raise loss. The gradient is information used to update w and b; **dw is not a trainable parameter**.

## Chain rule through activation

We now introduce a nonlinear activation:

~~~math
z=wx+b,\qquad a=\operatorname{ReLU}(z),\qquad L=(a-y)^2.
~~~

Read ∂L/∂z as "how the loss changes when z changes." z changes a, and a changes L, so the rates multiply **along this dependency**:

~~~math
\frac{\partial L}{\partial z}
=\frac{\partial L}{\partial a}
\frac{\partial a}{\partial z}.
~~~

- ∂L/∂a = 2(a − y) from the power rule applied to (a−y)^2.
- ∂a/∂z = ReLU'(z): 0 if z < 0; 1 if z > 0.
- At exactly z = 0, ReLU is not differentiable; using 0 is an implementation convention.

For z = −3, y = 2: a = 0, ∂L/∂a = −4, ∂a/∂z = 0, so ∂L/∂z = −4 * 0 = **0**. The local gradient is blocked at this ReLU.

A previously used "incoming gradient = 6" is just an **assumed illustration**, not a value derived from this particular squared-error example.

**Next:** extend w -> z -> a -> L to compute ∂L/∂w and ∂L/∂b, then combine layers. The formula has been introduced, but hidden-layer backpropagation is not yet coded.

## References

- [Neural-Network: full-batch implementation](https://github.com/Vaibh37/Neural-Network/blob/main/examples/train_batch_neuron_numpy.py)
- [Neural-Network: ReLU and chain-rule introduction](https://github.com/Vaibh37/Neural-Network/blob/main/notes/08-relu-derivative-and-chain-rule-intro.md)
