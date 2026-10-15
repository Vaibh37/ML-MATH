# 03 — Transpose and Matrix–Vector Multiplication

## Transpose swaps rows and columns

$$
\begin{bmatrix}1&2&3\\4&5&6\end{bmatrix}^{T}
=
\begin{bmatrix}1&4\\2&5\\3&6\end{bmatrix}.
$$

The shape changes from (2,3) to **(3,2)**.

## Matrix × column vector

A vector can be written as a one-column matrix in mathematics:

$$
W=\begin{bmatrix}2&3\\4&1\end{bmatrix},
\quad x=\begin{bmatrix}5\\2\end{bmatrix}
$$

Calculate one dot product per row:

$$
Wx=\begin{bmatrix}
2\cdot5+3\cdot2\\
4\cdot5+1\cdot2
\end{bmatrix}
=\begin{bmatrix}16\\22\end{bmatrix}.
$$

Shape: (2,2) × (2,1) → (2,1).

~~~python
import numpy as np
W = np.array([[2, 3], [4, 1]])
x = np.array([5, 2])   # One-dimensional NumPy array: shape (2,)
print(W @ x)          # [16 22], shape (2,)
print(W @ x[:, None]) # [[16], [22]], shape (2, 1)
print(W.T.shape)      # (2, 2)
~~~

**Be precise:** a NumPy array of shape (2,) and a two-dimensional column of shape (2,1) are not the same shape. This distinction matters for bias broadcasting.

Each row of W can represent one dense neuron; Wx contains its weighted sum before bias and activation.
