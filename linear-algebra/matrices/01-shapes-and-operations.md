# 01 — Matrix Shapes and Operations

A matrix has rows and columns. Its shape is **(rows, columns)**.

$$
A=\begin{bmatrix}2&4&6\\1&3&5\end{bmatrix},\quad A:(2,3).
$$

Nested lists describe **one two-dimensional array**, with one inner list per row:

~~~python
import numpy as np
A = np.array([[2, 4, 6], [1, 3, 5]])
print(A.ndim)   # 2
print(A.shape)  # (2, 3)
~~~

## Add and subtract matching positions

$$
\begin{bmatrix}2&4\\6&8\end{bmatrix}+
\begin{bmatrix}1&3\\5&7\end{bmatrix}
=\begin{bmatrix}3&7\\11&15\end{bmatrix}.
$$

$$
\begin{bmatrix}8&6\\4&2\end{bmatrix}-
\begin{bmatrix}3&1\\2&5\end{bmatrix}
=\begin{bmatrix}5&5\\2&-3\end{bmatrix}.
$$

## Multiply by a scalar

$$
-2\begin{bmatrix}2&-3\\4&5\end{bmatrix}
=\begin{bmatrix}-4&6\\-8&-10\end{bmatrix}.
$$

**Important:** Ordinary matrix addition expects equal shapes. NumPy supports some unequal-shape additions through **broadcasting**, covered later.

**Practice:** Find the shape of a matrix with 2 rows and 4 columns. Then verify the two sums above with NumPy.
