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

```{python, echo=False, eval=True}
def f(x):
    return x ** 2

f(2)
```

Don't evaluate.

```{python, eval=False}
def f(x):
    return x ** 2

f(2)
```

## Plotting

```{python}
%matplotlib inline
import matplotlib.pyplot as plt
plt.plot(range(4), range(4))
```

```{python}
print("2 + 2 is")
2 + 2
```

Fin.

