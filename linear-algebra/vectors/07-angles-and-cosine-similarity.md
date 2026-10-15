# 07 — Vector Angles and Cosine Similarity

**Status:** Practiced on 14–15 October 2026. **Prerequisites:** dot product and magnitude.

The dot product connects two nonzero vectors to the angle between them:

$$
\mathbf a\cdot\mathbf b=\|\mathbf a\|\|\mathbf b\|\cos\theta
$$

$$
\cos\theta=\frac{\mathbf a\cdot\mathbf b}{\|\mathbf a\|\|\mathbf b\|},\qquad
\theta=\arccos(\cos\theta).
$$

The angle is between 0° and 180°. The formula is undefined if either vector has zero magnitude.

| Cosine similarity | Angle | Meaning |
|---:|---:|---|
| 1 | 0° | Same direction |
| 0 | 90° | Perpendicular |
| -1 | 180° | Opposite directions |

## Worked example: perpendicular vectors

For a = [3, 4] and b = [-4, 3]:

1. Dot product: (3 × -4) + (4 × 3) = 0.
2. Magnitudes: 5 and 5.
3. Cosine similarity: 0 / (5 × 5) = 0.
4. Angle: arccos(0) = **90°**.

## Worked example: direction matters

The vectors [1, 0] and [-1, 0] have angle **180°**. The vectors [-1, 0] and [-1, 0] have angle **0°**. Read both vectors before computing.

## NumPy verification

~~~python
import math
import numpy as np

a = np.array([3.0, 4.0])
b = np.array([-4.0, 3.0])
denominator = np.linalg.norm(a) * np.linalg.norm(b)
if denominator == 0:
    raise ValueError("Angle is undefined for a zero vector.")

cosine = np.dot(a, b) / denominator
angle = math.degrees(math.acos(float(np.clip(cosine, -1.0, 1.0))))
print(cosine)  # 0.0
print(angle)   # 90.0
~~~

Clipping to [-1, 1] protects arccos from small floating-point rounding errors.

**ML connection:** cosine similarity is used to compare directions of embedding vectors. It is **not** itself a probability.

Related: [trigonometry tables](../../trigonometry/TRIG_CHEATSHEET.md).
