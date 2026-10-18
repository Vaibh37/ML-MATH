# 04 — Calculus Behind a Two-Layer XOR Neural Network (18 October 2026)

**Status:** Applied the chain rule to a working **2 → 3 → 1 NumPy network** with ReLU hidden units, sigmoid output, mean squared error, and full-batch gradient descent. This extends [03 — batch MSE and ReLU](03-batch-mse-and-relu-chain-rule.md).

**Executable implementation and the high-resolution visual series:** [Neural-Network XOR example](https://github.com/Vaibh37/Neural-Network/blob/main/examples/xor_network.py) · [12 diagrams](https://github.com/Vaibh37/Neural-Network/tree/main/assets/xor).

## 1. Symbols, dimensions, and forward equations

| Symbol | Definition | Shape when N = 4 |
| --- | --- | --- |
| X | Four XOR training inputs | (4, 2) |
| W_h, b_h | Hidden weights and biases | (2, 3), (3,) |
| Z_h | Hidden weighted sums | (4, 3) |
| H | ReLU hidden activations | (4, 3) |
| W_o, b_o | Output weights and bias | (3, 1), (1,) |
| Z_o | Output weighted sum | (4, 1) |
| A_o | Sigmoid outputs | (4, 1) |
| Y | Targets | (4, 1) |
| L | Mean squared error | scalar |

```math
Z_h=XW_h+b_h,\qquad H=\mathrm{ReLU}(Z_h)
```

```math
Z_o=HW_o+b_o,\qquad A_o=\sigma(Z_o)
```

```math
L=\frac1N\sum_{i=1}^N\big(a_o^{(i)}-y^{(i)}\big)^2
```

The computational path is **X → Z_h → H → Z_o → A_o → L**. Backpropagation follows it in the opposite direction.

[Visual: forward propagation and tensor shapes](https://github.com/Vaibh37/Neural-Network/blob/main/assets/xor/d04-forward-flow.png)

## 2. Derive the output pre-activation gradient

For one prediction, differentiating the square gives `2(a_o-y)`. Since L averages over N rows:

```math
\frac{\partial L}{\partial A_o}=\frac{2}{N}(A_o-Y).
```

Sigmoid satisfies:

```math
\sigma'(z)=\sigma(z)(1-\sigma(z)).
```

Apply the chain rule with respect to the **pre-activation** `Z_o`:

```math
\underbrace{\frac{\partial L}{\partial Z_o}}_{(N,1)}
=
\underbrace{\frac{\partial L}{\partial A_o}}_{(N,1)}
\odot
\underbrace{A_o\odot(1-A_o)}_{(N,1)}.
```

In NumPy:

```python
dl_da_o = 2 / len(X) * (a_o - y)
da_od_zo = a_o * (1 - a_o)
dl_dz_o = dl_da_o * da_od_zo
```

`\odot` means elementwise multiplication. We are **not** performing matrix multiplication in this line.

## 3. Output parameter gradients

For one sample:

```math
z_o=h_1v_1+h_2v_2+h_3v_3+b_o.
```

Therefore `\partial z_o/\partial v_j=h_j`, and `\partial z_o/\partial b_o=1`. Chain rule and summing the full batch yield:

```math
\frac{\partial L}{\partial W_o}=H^T\frac{\partial L}{\partial Z_o},
\qquad
\frac{\partial L}{\partial b_o}=\sum_{i=1}^N\frac{\partial L}{\partial z_o^{(i)}}.
```

```python
dW_o = h.T @ dl_dz_o
db_o = np.sum(dl_dz_o, axis=0)
```

The shapes are `(3,4) @ (4,1) → (3,1)` and `(1,)`.

**Do not divide by N again:** `dl_dz_o` already contains `2/N`.

## 4. Backpropagation into the hidden layer

An output pre-activation depends on each hidden activation through the **output weights**:

```math
\frac{\partial z_o}{\partial h_j}=v_j.
```

So for all samples and all hidden units:

```math
\frac{\partial L}{\partial H}=
\frac{\partial L}{\partial Z_o}W_o^T.
```

```python
dl_dh = dl_dz_o @ W_o.T
```

**Use `W_o.T`, not `dW_o.T`.** A weight tells us how the output depends on a hidden activation; a *weight gradient* tells us how the loss depends on that weight. These are different derivatives.

For ReLU, use the subgradient convention 0 at zero:

```math
\mathrm{ReLU}'(z)=
\begin{cases}
1,&z>0\\
0,&z\leq0.
\end{cases}
```

```math
\frac{\partial L}{\partial Z_h}
=\frac{\partial L}{\partial H}\odot\mathrm{ReLU}'(Z_h).
```

```python
dh_dhz = (z_h > 0).astype(float)
dl_dzh = dl_dh * dh_dhz
```

ReLU masks negative hidden pre-activations. If a unit is inactive for a row, its local gradient on that row is zero.

## 5. Hidden weight and bias gradients

Each hidden pre-activation depends linearly on its input features:

```math
z_{h,j}^{(i)}
=x_1^{(i)}w_{1j}+x_2^{(i)}w_{2j}+b_{h,j}.
```

Therefore `\partial z_{h,j}^{(i)}/\partial w_{kj}=x_k^{(i)}`, and `\partial z_{h,j}^{(i)}/\partial b_{h,j}=1`.

For the entire batch:

```math
\frac{\partial L}{\partial W_h}
=X^T\frac{\partial L}{\partial Z_h},
\qquad
\frac{\partial L}{\partial b_h}
=\sum_{i=1}^{N}\frac{\partial L}{\partial z_h^{(i)}}.
```

```python
dW_h = X.T @ dl_dzh
db_h = np.sum(dl_dzh, axis=0)
```

Shapes: `(2,4) @ (4,3) → (2,3)`, and `(3,)`.

## 6. Why the four updates happen together

Once all four gradients are computed using the **same current model**:

```math
\theta_{\mathrm{new}}=\theta-\alpha\frac{\partial L}{\partial\theta}.
```

```python
W_h -= A * dW_h
b_h -= A * db_h
W_o -= A * dW_o
b_o -= A * db_o
```

This is **full-batch gradient descent**: one gradient per parameter averaged across the four examples, followed by one update of each parameter per epoch.

**Completed:** 13 parameter gradients were implemented, and a seeded 5,000-epoch XOR training run correctly classified all four examples with MSE `0.00471529585312808`. **Not claimed:** a full finite-difference check of all 13 gradients; that verification was deliberately deferred.

For the full sequence with explanatory diagrams, see [Neural-Network lessons 11–16](https://github.com/Vaibh37/Neural-Network/tree/main/notes).
