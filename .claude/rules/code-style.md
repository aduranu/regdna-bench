# Python Style Rules

These rules are mandatory. Follow them exactly when writing or editing Python code.

---

## 1. Indentation & Spacing

- Use 4 spaces for indentation (never tabs).
- Add **blank lines between logical blocks** inside functions:
  - setup
  - main action
  - return
- Use:
  - **1 blank line** between top-level functions
  - **2 blank lines** between major sections/functions

- No trailing whitespace.
- Keep dicts/lists **compact when readable**.

---

## 2. Function Structure

- **only use nested function definitions when absolutely necessary**
  - Use top-level helpers instead.
- Try to keep functions flat and readable.

---

## 3. Comments (High Signal Only)

Write comments only for **context** and description of **non obvious** functions/loops.

### Rules:
- Explain **why** or **where this is used**, not what the code does
- Write like a quick note to a colleague
- Use **lowercase**, direct tone
- Avoid redundancy with the function name

### Example:

GOOD:
```python
# used by concurrency/variant sweeps where multiple variants share one large interval (2^17 bp)
generate_variants_in_interval(interval, n, spacing=100, offset=1000)
```

BAD:
```python
generate_variants_in_interval(interval, n, spacing=100, offset=1000)
"""Generate n variants placed within the given interval."""
```

---

## 4. Printing

- NO emojis
- NO characters like this: →
- Only print when processes are FINISHED, not when they are starting
- Errors should always be properly raised, not printed
- Be conservative with error raising unless explicitly instructed to do the opposite

---

## 5. Private / Public Functions

- Public APIs: no leading underscore.
- Internal helpers: single leading underscore.

GOOD:
```python
def train_model(...):
    ...
```

```python
def _load_checkpoint(...):
    ...
```

---

## 6. Abstract Classes

Use abstract base classes for core interfaces:
- DataLoader
- Dataset
- Model
- Trainer
- Storage backend

Rules:
- Use `ABC` and `@abstractmethod`.
- Define interfaces, not implementations.
- Keep interfaces minimal.
- Prefer composition over deep inheritance.

---

## 7. Numerical Precision in Machine Learning

Precision decisions must be explicit.

Rules:
- Define training/inference/eval dtypes near the top of the file.
- Do not rely on framework defaults.
- Cast explicitly when precision matters.
- If using defaults, document them.
- Checkpoint loading should explicitly preserve or cast precision.
- Metrics and reductions should use intentional precision.

GOOD:
```python
TRAIN_DTYPE = torch.bfloat16
INFERENCE_DTYPE = torch.float16
METRIC_DTYPE = torch.float32
```
