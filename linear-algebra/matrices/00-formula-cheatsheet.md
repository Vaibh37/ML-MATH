# Matrix Formula Cheatsheet

Let A have shape (m, n) and B have shape (n, p).

| Operation | Requirement | Result shape |
|---|---|---|
| Addition / subtraction | Same shape for ordinary matrix arithmetic | (m, n) |
| Scalar multiplication | One scalar | (m, n) |
| Matrix multiplication A × B | A columns = B rows | (m, p) |
| Transpose Aᵀ | None | (n, m) |
| Matrix × column vector | Vector length = n | (m, 1) |

Matrix product entry: (AB)ᵢⱼ = sum over k of Aᵢₖ × Bₖⱼ.

**Full example:**

$$
\begin{bmatrix}1&2\\3&4\end{bmatrix}
\begin{bmatrix}5&6\\7&8\end{bmatrix}
=
\begin{bmatrix}19&22\\43&50\end{bmatrix}.
$$

## NumPy code

~~~python
import numpy as np
A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])

print(A + B)   # [[6, 8], [10, 12]]
print(A * B)   # [[5, 12], [21, 32]]: elementwise!
print(A @ B)   # [[19, 22], [43, 50]]: matrix product
print(A.T)     # [[1, 3], [2, 4]]: transpose
~~~

## Broadcasting is a separate NumPy rule

Compare shapes from right to left. Each pair must be equal **or** one must be 1. Missing leading dimensions behave like 1.

| Addition shapes | Valid? | Output |
|---|---|---|
| (3, 4) + (3, 1) | Yes | (3, 4) |
| (3, 4) + (1, 4) | Yes | (3, 4) |
| (3, 1, 5) + (1, 4, 5) | Yes | (3, 4, 5) |
| (3, 4) + (2, 4) | No | Error |

For column-oriented sample batches, W:(m,n), X:(n,s), b:(m,1) gives **WX + b:(m,s)**. Bias is reused for each of the s input columns.

Many ML libraries instead store samples in rows. Do not mix batch conventions.
