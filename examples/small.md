---
title: small
author: test author
---

# This is a small example

```{python}
def f(x):
    return x * 2

f(2)
```

With options

```{.python echo=False eval=True}
def f(x):
    return x ** 2

f(2)
```

Don't evaluate.

```{.python eval=False}
def f(x):
    return x ** 2

f(2)
```

Fin.

