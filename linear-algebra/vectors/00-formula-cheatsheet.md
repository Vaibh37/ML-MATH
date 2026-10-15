# Vector Formula Cheatsheet

Let $\mathbf a=[a_1,a_2,\ldots,a_n]$, $\mathbf b=[b_1,b_2,\ldots,b_n]$, and let $k,m$ be **scalars** (ordinary numbers).

| Operation | General formula | Example | Output |
|---|---|---|---|
| Addition | $\mathbf a+\mathbf b=[a_1+b_1,\ldots,a_n+b_n]$ | $[3,4]+[1,2]=[4,6]$ | Vector |
| Subtraction | $\mathbf a-\mathbf b=[a_1-b_1,\ldots,a_n-b_n]$ | $[7,5]-[2,3]=[5,2]$ | Vector |
| Scalar multiplication | $k\mathbf a=[ka_1,\ldots,ka_n]$ | $3[3,4,2]=[9,12,6]$ | Vector |
| Linear combination | $k\mathbf a+m\mathbf b=[ka_1+mb_1,\ldots,ka_n+mb_n]$ | $2[1,2]+3[3,4]=[11,16]$ | Vector |
| Dot product | $\mathbf a\cdot\mathbf b=\sum_{i=1}^{n}a_i b_i$ | $[3,4]\cdot[1,2]=3+8=11$ | Scalar |
| Magnitude | $\|\mathbf a\|=\sqrt{a_1^2+\cdots+a_n^2}$ | $\|[2,3]\|=\sqrt{13}$ | Scalar |
| Unit vector | $\hat{\mathbf a}=\frac{\mathbf a}{\|\mathbf a\|}$ for $\mathbf a\ne\mathbf0$ | $[1,2]\to[1/\sqrt5,2/\sqrt5]$ | Unit vector |

## Magnitude by dimension

**2D:** $\|[x,y]\|=\sqrt{x^2+y^2}$.

**3D:** $\|[x,y,z]\|=\sqrt{x^2+y^2+z^2}$.

**Any dimension:** $\|\mathbf a\|=\sqrt{\sum_i a_i^2}$.

## How to prove a vector is a unit vector

For $\mathbf a=[1,2]$:

$$
\|\mathbf a\|=\sqrt{1^2+2^2}=\sqrt5
$$

Normalize:

$$
\hat{\mathbf a}=\left[\frac1{\sqrt5},\frac2{\sqrt5}\right]
$$

**Square the components** when verifying:

$$
\|\hat{\mathbf a}\|=\sqrt{\left(\frac1{\sqrt5}\right)^2+\left(\frac2{\sqrt5}\right)^2}
=\sqrt{\frac15+\frac45}=\sqrt1=1
$$

**Common mistake:** $1/\sqrt5+2/\sqrt5$ is **not** the magnitude.

## Geometry formulas (reviewed)

$\mathbf a\cdot\mathbf b=\|\mathbf a\|\|\mathbf b\|\cos\theta$.

$\cos\theta=\dfrac{\mathbf a\cdot\mathbf b}{\|\mathbf a\|\|\mathbf b\|}$ for two nonzero vectors.

These geometric interpretations were practiced with 0°, 90°, and 180° examples. See [vector angles](07-angles-and-cosine-similarity.md).

## NumPy reference

~~~python
import numpy as np
a = np.array([3, 4])
b = np.array([1, 2])

print(a + b)                  # [4 6]
print(a - b)                  # [2 2]
print(3 * a)                  # [ 9 12]
print(np.dot(a, b))           # 11
print(np.linalg.norm(a))      # 5.0
print(a / np.linalg.norm(a))  # [0.6 0.8]
~~~
