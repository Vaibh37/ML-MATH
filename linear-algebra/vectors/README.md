# Vectors for Machine Learning

A practical collection of the vector lessons studied on 14 October 2026. Each lesson includes the mathematical rule, a worked example, a short NumPy implementation, and a diagram.

Start with the [formula cheatsheet](00-formula-cheatsheet.md), then follow the lessons in order.

| Lesson | Topic | Status |
|---|---|---|
| [01](01-vector-basics.md) | Coordinates, components, dimensions | Reviewed |
| [02](02-addition-and-subtraction.md) | Vector addition and subtraction | Practiced |
| [03](03-scalar-multiplication-and-linear-combinations.md) | Scaling and combining vectors | Practiced / introduced |
| [04](04-dot-product.md) | Dot product and scalar output | Practiced |
| [05](05-vector-magnitude.md) | Euclidean length in 2D and 3D | Practiced |
| [06](06-unit-vectors.md) | Normalization and checking length = 1 | Practiced |

**Existing introductory lesson:** [Vectors: CS perspective](../01-vectors-introduction.md).

## Diagram files

- The rendered **SVG** illustrations appear in the lessons and display directly on GitHub. They stay sharp at any resolution.
- Matching **.excalidraw** files are included in [assets/](assets/) as editable Excalidraw scene sources. Open them at [excalidraw.com](https://excalidraw.com) via Open > Load from file.
- A PNG export bundle is supplied separately for download; SVG is the canonical image version within GitHub.

## Important distinctions

1. Vector addition and subtraction give **vectors**.
2. The dot product gives a **scalar** (number), not a one-element vector.
3. The unit vector's **components** are not generally 1; its **total magnitude** is 1.
4. Normalize a nonzero vector only. The zero vector has no direction and cannot be normalized.

## Next concepts (not completed yet)

Angles between vectors, cosine similarity, projection, span, linear independence, and matrices.
