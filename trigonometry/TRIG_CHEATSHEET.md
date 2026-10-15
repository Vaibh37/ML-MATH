# Trigonometry Cheat Sheet: sin, cos, tan and Inverse Trig

Quick reference for exact values in **degrees**. Useful for vectors, dot products, cosine similarity and machine learning mathematics.

![Normal and inverse trigonometry tables](https://raw.githubusercontent.com/Vaibh37/ML-MATH/main/trigonometry/assets/trigonometry_normal_inverse.png)

## 1. Normal trigonometric values

| Angle (θ) | sin θ | cos θ | tan θ |
|---|---:|---:|---:|
| 0° | 0 | 1 | 0 |
| 30° | 1/2 | √3/2 | 1/√3 |
| 45° | √2/2 | √2/2 | 1 |
| 60° | √3/2 | 1/2 | √3 |
| 90° | 1 | 0 | Undefined |
| 180° | 0 | −1 | 0 |
| 270° | −1 | 0 | Undefined |
| 360° | 0 | 1 | 0 |

- **tan θ = sin θ / cos θ**, so tangent is undefined where cos θ = 0.
- **sin² θ + cos² θ = 1**.
- Some books use √2/2 in place of 1/√2, and √3/3 in place of 1/√3. These are equal.

## 2. Inverse sine: arcsin(x) or sin⁻¹(x)

| Input x | arcsin(x) |
|---|---:|
| −1 | −90° |
| −√3/2 | −60° |
| −√2/2 | −45° |
| −1/2 | −30° |
| 0 | 0° |
| 1/2 | 30° |
| √2/2 | 45° |
| √3/2 | 60° |
| 1 | 90° |

Real input domain: **−1 ≤ x ≤ 1**. Principal output range: **−90° to 90°**.

## 3. Inverse cosine: arccos(x) or cos⁻¹(x)

| Input x | arccos(x) |
|---|---:|
| −1 | 180° |
| −√3/2 | 150° |
| −√2/2 | 135° |
| −1/2 | 120° |
| 0 | 90° |
| 1/2 | 60° |
| √2/2 | 45° |
| √3/2 | 30° |
| 1 | 0° |

Real input domain: **−1 ≤ x ≤ 1**. Principal output range: **0° to 180°**.

## 4. Inverse tangent: arctan(x) or tan⁻¹(x)

| Input x | arctan(x) |
|---|---:|
| −√3 | −60° |
| −1 | −45° |
| −1/√3 | −30° |
| 0 | 0° |
| 1/√3 | 30° |
| 1 | 45° |
| √3 | 60° |

Real input domain: **all real numbers**. Principal output range: **strictly between −90° and 90°**.

## 5. Essential distinctions

- **sin⁻¹(x) means arcsin(x)**, *not* 1/sin(x). Likewise for cos⁻¹ and tan⁻¹. For reciprocals, write csc(x), sec(x), cot(x), or 1/sin(x).
- The inverse functions return their **principal angle**, not every angle with that sine, cosine or tangent.
- Example: sin(150°) = 1/2, but **arcsin(1/2) = 30°**, not 150°.
- Example: tan(150°) = −1/√3, but **arctan(−1/√3) = −30°**, not 150°.

## 6. Why it matters for vectors and ML

For nonzero vectors a and b, their angle is determined by

\[
\cos(\theta) = \frac{\mathbf{a}\cdot\mathbf{b}}{\lVert\mathbf{a}\rVert\,\lVert\mathbf{b}\rVert}
\]

Thus, **θ = arccos(cosine similarity)**. For example, a cosine similarity of **0** corresponds to **90°**, and a similarity of **−1** corresponds to **180°**.

### Python quick check

```python
import math

print(math.sin(math.radians(30)))          # approximately 0.5
print(math.degrees(math.acos(0)))          # 90.0
print(math.degrees(math.asin(0.5)))        # approximately 30.0
print(math.degrees(math.atan(1)))          # 45.0
```

Python's `math.sin`, `math.cos`, `math.tan` and inverse trig functions **use and return radians**. Convert with `math.radians` and `math.degrees` for degree-based work.

**Learning order:** Memorize 0°, 30°, 45°, 60°, 90° first, then extend to quadrants and negative values. Read the **principal ranges** before using inverse trig.
