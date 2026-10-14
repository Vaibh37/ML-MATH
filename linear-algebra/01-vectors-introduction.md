# 01 — Vectors: Introduction (Computer Science Perspective)

**Status:** Video watched on 14 October 2026; examples available for independent practice.  
**Source:** [3Blue1Brown — Vectors | Essence of Linear Algebra, Chapter 1](https://youtu.be/fNk_zzaMoSs)  
**Focus:** What a vector means when it is represented, stored, and manipulated in code.

> These notes summarize the video's **computer science perspective** and connect it to NumPy/ML. They deliberately skip physical motion, force diagrams, and formal abstract vector-space theory. The ML examples are applications of the ideas, not topics taught in the video.

## 1. What is a vector in CS?

A vector is an **ordered collection of numbers**. Its entries are called *components*. The order and meaning of the components matter.

```python
import numpy as np

features = np.array([25, 175, 70])

print(features)        # [ 25 175  70]
print(features.shape)  # (3,)
print(features.ndim)   # 1
```

This could represent a record with three features: age, height, and weight (the real-world values and units are only examples).

- **Order matters:** `[age, height, weight]` is not the same representation as `[weight, age, height]`.
- **Dimension matters:** The vector above has three components. In NumPy, a 1D array holding three components has shape `(3,)`.
- **Meaning matters:** Adding unrelated features is not automatically useful merely because NumPy allows it.

**Vector vs Python list:** Python lists can hold mixed objects. A NumPy array is a useful representation for numeric vector operations, usually with one common dtype.

## 2. From coordinates to ordered values

The video shows that coordinates give a numerical description of a vector. For CS, the key point is that **we can store those coordinates as an array** and operate on them without needing a drawing.

```python
v2 = np.array([3, 2])       # two components
v3 = np.array([3, 2, -1])   # three components
v5 = np.array([1, 0, 4, 2, 7])

print(v2.shape)  # (2,)
print(v3.shape)  # (3,)
print(v5.shape)  # (5,)
```

An ML vector may have **hundreds or thousands of components**. We do not need a geometric picture of 1,000-dimensional space to store it or compute with it.

A useful mental model: each position in the vector is a named slot, even if the code stores only numbers.

```python
feature_names = ["age", "height_cm", "weight_kg"]
x = np.array([25, 175, 70])

print(dict(zip(feature_names, x.tolist())))
# {'age': 25, 'height_cm': 175, 'weight_kg': 70}
```

## 3. Vector addition = componentwise addition

To add vectors, add **matching positions**. For the introductory case, they should have the same number of components and represent compatible quantities.

```python
a = np.array([2, 3])
b = np.array([4, 1])

result = a + b
print(result)  # [6 4]
```

What happens position by position?

| Position | `a` | `b` | Sum |
|---|---:|---:|---:|
| index 0 | 2 | 4 | 6 |
| index 1 | 3 | 1 | 4 |

A simple CS example: combine two numeric changes to the *same* measurements.

```python
baseline = np.array([10, 20, 30])
delta = np.array([1, -2, 4])
print(baseline + delta)  # [11 18 34]
```

**Important distinction:** NumPy has *broadcasting*, so arrays with some different shapes can still be added. But broadcasting is a NumPy feature; it is **not** a reason to treat two different-sized feature vectors as interchangeable.

## 4. Scalar multiplication = scale every component

A **scalar** is one number. Multiplying a vector by a scalar multiplies **each component** by that number.

```python
v = np.array([2, -3])

print(2 * v)     # [ 4 -6]
print(0.5 * v)   # [ 1.  -1.5]
print(0 * v)     # [0 0]
print(-1 * v)    # [-2  3]
```

In code, think: **one numeric multiplier applied to each element**. This operation appears in weighted numerical calculations and parameter updates.

## 5. Vector addition and scaling together

We can combine both operations to create a new vector:

```python
a = np.array([1, 2])
b = np.array([3, 4])

combined = 2 * a + 3 * b
print(combined)  # [11 16]
```

Calculation by components:
- Index 0: `2 * 1 + 3 * 3 = 11`
- Index 1: `2 * 2 + 3 * 4 = 16`

This is the coding intuition behind **combining vectors with numeric weights**. The phrase *linear combination*, and why it matters, can be covered in the next linear-algebra lessons.

## 6. What this enables in ML

These are **CS/ML connections** to the chapter's ideas, rather than claims that the video directly teaches ML:

| Representation | What the vector's components mean |
|---|---|
| Feature vector | Measurements or encoded properties of one training example |
| Model parameter vector | A set of numeric weights |
| Word/document embedding | Learned numeric representation of text |
| Batch of feature vectors | Multiple examples, often stored as rows of a 2D array |

```python
# One example, represented by three features
x = np.array([0.2, 0.5, 0.9])     # shape (3,)

# Three examples, each with three features
X = np.array([
    [0.2, 0.5, 0.9],
    [0.1, 0.7, 0.4],
    [0.6, 0.3, 0.8],
])
print(X.shape)  # (3, 3): three examples x three features
```

Do not confuse the number of **samples** with the number of **features**: for an `X` with shape `(100, 3)`, there are 100 sample rows and 3 feature columns.

In real ML pipelines, features can use different units or scales. We will learn preprocessing/normalization separately.

## 7. What we should remember

1. A vector in CS is an **ordered set of numeric components**.
2. The position of each component has meaning.
3. Its *dimension* is the number of components, not the same thing as NumPy's `.ndim`. Example: a 5-component vector is represented as a **1D** NumPy array with shape `(5,)`.
4. Vector addition combines matching components.
5. Multiplying by a scalar scales every component.
6. NumPy lets us implement these operations directly.
7. Vectors become ML feature representations, parameters, and embeddings.

### Common mistakes to avoid

| Mistake | Correction |
|---|---|
| A 3-component vector must be a 3D NumPy array | `np.array([1, 2, 3])` has shape `(3,)` and `ndim == 1`. |
| Vector addition combines any two lists regardless of meaning | Components must align in position and meaning. |
| `2 * [1, 2]` behaves like `2 * np.array([1, 2])` | Python list repetition gives `[1, 2, 1, 2]`; NumPy scales to `[2, 4]`. |
| We must visualize every feature vector as an arrow | High-dimensional ML representations can be used numerically. |
| This video covered dot products or cosine similarity | Those are **next topics**, not part of this introduction. |

## 8. Tiny self-check (write code, do not copy answers)

1. Create a NumPy vector with 4 values. Print its `shape` and `ndim`.
2. Add `[2, 5, -1]` and `[3, -2, 4]` by hand, then verify with NumPy.
3. Multiply `[4, -2, 6]` by `-0.5`. Predict the output first.
4. Explain in one sentence why `[10, 20, 30]` and `[30, 10, 20]` are different feature representations when positions have fixed meanings.

## Next lesson

**Dot product → vector magnitude → cosine similarity (ML embeddings)**, followed by matrices and matrix multiplication.

These concepts will connect vector operations to neural-network weighted sums and embedding search.

**One-line takeaway:** In CS, vectors are structured numeric data; addition and scalar scaling are the first tools for transforming them.
