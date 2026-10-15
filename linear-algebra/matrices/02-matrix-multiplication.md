# 02 — Matrix Multiplication: Row × Column

Take **one row from A**, **one column from B**, multiply matching entries, and add. Repeat for every output position.

![Matrix multiplication diagram](assets/matrix-multiplication.svg)

For A:(m,n) and B:(n,p), the inner dimensions match and the product has shape **(m,p)**.

## Worked example: calculate all four entries

$$
A=\begin{bmatrix}1&2\\3&4\end{bmatrix},\quad
B=\begin{bmatrix}5&6\\7&8\end{bmatrix}
$$

1. Top-left: (1 × 5) + (2 × 7) = **19**.
2. Top-right: (1 × 6) + (2 × 8) = **22**.
3. Bottom-left: (3 × 5) + (4 × 7) = **43**.
4. Bottom-right: (3 × 6) + (4 × 8) = **50**.

$$
AB=\begin{bmatrix}19&22\\43&50\end{bmatrix}
$$

## Rectangular example

$$
A=\begin{bmatrix}1&2&3\\4&5&6\end{bmatrix}:(2,3),\quad
B=\begin{bmatrix}7&8\\9&10\\11&12\end{bmatrix}:(3,2)
$$

$$
AB=\begin{bmatrix}58&64\\139&154\end{bmatrix}:(2,2).
$$

## Python

~~~python
import numpy as np
A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])
print(A * B)  # [[5, 12], [21, 32]]: elementwise
print(A @ B)  # [[19, 22], [43, 50]]: matrix product
~~~

The @ operator is **not** the same as *. Matrix multiplication is not generally commutative: AB can differ from BA.

**Practice:** What shape results from (3,2) × (2,5)? Answer: (3,5).
