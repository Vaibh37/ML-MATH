# 04 — NumPy Broadcasting and Bias

Broadcasting handles certain **elementwise array operations**, including addition with different shapes. It does **not** change the matrix multiplication rule.

## Rule: check every dimension

Compare shapes from the right. Each pair must be **equal**, or **at least one equals 1**. Leading missing dimensions behave like 1.

Examples:
- (3,4) + (3,1) → (3,4), valid.
- (3,4) + (1,4) → (3,4), valid.
- (3,1,5) + (1,4,5) → (3,4,5), valid.
- (3,4) + (2,4) → invalid because 3 and 2 conflict.

![Broadcasting diagram](assets/broadcasting.svg)

## Worked example: repeat a row (conceptually)

$$
A=\begin{bmatrix}5&10\\15&20\\25&30\end{bmatrix}:(3,2),
\quad B=\begin{bmatrix}2&3\end{bmatrix}:(1,2)
$$

B behaves as if it were:

$$
B_{\mathrm{expanded}}=\begin{bmatrix}2&3\\2&3\\2&3\end{bmatrix}.
$$

Therefore,

$$
A+B=\begin{bmatrix}7&13\\17&23\\27&33\end{bmatrix}:(3,2).
$$

## Worked example: dense-layer bias

$$
WX=\begin{bmatrix}19&22\\43&50\end{bmatrix}:(2,2),
\quad b=\begin{bmatrix}1\\2\end{bmatrix}:(2,1).
$$

The bias is used for **both input columns**:

$$
WX+b=
\begin{bmatrix}19+1&22+1\\43+2&50+2\end{bmatrix}
=\begin{bmatrix}20&23\\45&52\end{bmatrix}.
$$

The result stays shape (2,2). NumPy can perform this without physically duplicating all the bias values.

~~~python
import numpy as np
WX = np.array([[19, 22], [43, 50]])
b = np.array([[1], [2]]) # (2,1), correct for sample columns
print(WX + b)
# [[20 23]
#  [45 52]]
~~~

**Common mistake:** using a NumPy 1D bias of shape (2,) would line up the *last* dimension and produce different values in this example. Use (number_of_neurons, 1) for column-oriented batches.

**Practice:** explain why (3,4)+(2,4) fails and why (3,4)+(3,1) works.
