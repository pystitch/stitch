---
header-includes:
    - \usepackage{booktabs}
---
# Usage

This document will display some of `stitch`'s options.


## Code-Chunks

The syntax for creating an executable block is

    ```{kernel_name}
    # Your code here
    ```

The output of that block, if any, will be inserted just after the end of the
code block[^literal_code].

Code chunks can accept up to two positional arguments

- `kernel_name` (required)
  See `jupyter-kernelspect list` for a list of the kernels installed on your system
- `chunk_name` (optional)
  And identifier for the code chunk. Used in several places for naming figures and files

And some keyword arguments

- `echo`
- `eval`
- `include`?

The rest of this document is intended to demonstrate `stitch` in action.
View the markdown source alongside the rendered HTML.

---

## Basics

`stitch` keeps a registry of kernels, meaning state is preserved between
code chunks

```{python}
x = 10
```


There was no output there, but we can reuse `x` now

```{python}
print(x + 2)
```

Now we see the output.

---

## Exceptions

By default, exceptions are displayed and the stitching continues

```{python}
raise ValueError("ValueError!")
```

---

## Rich Display

We reuse IPython's [rich display system](http://ipython.readthedocs.io/en/stable/config/integrating.html),
so objects defining `_repr_html_`, `_repr_latex_`, etc. will have that
represented in the output.
Pandas DataFrames, for example, do so

```{python}
import pandas as pd
pd.options.display.latex.repr = True
import seaborn as sns
df = sns.load_dataset("iris")
df.head()
```


## Graphics

It's possible to capture rich output, like graphics


```{python}
%matplotlib inline
%config InlineBackend.figure_format = 'svg'

sns.set()
sns.pairplot(df, hue="species");
```

[^literal_code]: If you look at the markdown source for this document,
you'll see that I've indented the code block by 4 spaces. This is so that
the block is interpreted as a literal chunk of text, and isn't intercepted
by the engine for execution. To actually run code the code chunk should be at
the start of the line.
