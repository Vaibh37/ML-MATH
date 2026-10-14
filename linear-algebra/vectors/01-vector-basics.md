# 01 — Vector Basics

A **vector** can be represented in programming as an ordered collection of numbers. Each entry is a **component**. In a geometric view, $[x,y]$ points from the origin to $(x,y)$.

## Reading a vector

- $[3,4]$ has 2 components (2D).
- $[1,3,2]$ has 3 components (3D).
- $[2,0,-1,4]$ has 4 components (4D).

The vector $[3,4]$ means 3 units in the $x$ direction and 4 units in the $y$ direction. It does **not** mean a scalar value of 7.

~~~python
import numpy as np
a = np.array([3, 4])
print(a.shape)  # (2,) — two components
print(a.ndim)   # 1 — the NumPy array itself is one-dimensional
print(a.size)   # 2 — number of components
~~~

**Distinction:** A mathematical 2D vector is often stored in a NumPy 1D array of shape $(2,)$. Mathematical vector dimension and NumPy's array-axis count are different ideas.

In ML, a feature vector such as $[25,175,70]$ can represent a row of three numeric features. Coordinate order matters.

For a deeper introduction, read the [original CS-perspective lesson](../01-vectors-introduction.md).

## Quick check

What are the number of components and NumPy shape for $[4,-2,1]$?

**Answer:** 3 components and shape $(3,)$.
