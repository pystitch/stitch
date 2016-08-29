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

sns.set()
sns.pairplot(df, hue="species");
```

You can control image attributes from the chunk options

```{python, width=80, height=80px}
%matplotlib inline

sns.set()
sns.pairplot(df, hue="species");
```

## Document Options

You can provide document-wide options in a YAML metadata block at the
start of the file.
This looks like

    ```
    header-includes:
        - \usepackage{booktabs}
    ```


See the [pandoc documentation](http://pandoc.org/MANUAL.html) about the `yaml_metadata_block` extension for all the options.
In addition to Pandoc's options, `stitch` defines and intercepts the following
variables

## Command-line Interface

The command-line interface is as similar to pandoc's as possible.
The simplest example is just passing an input markdown file:

    ```
    stitch input.md
    ```

This will take the text in `input.md` execute it and convert it to HTML, printing the result to stdout.

Other useful options are

- `-o` or `--output_file`: the file to write the stitched output to; defaults to stdout.
- `-t` or `--to`: the type to transform to. Defaults to `html` or it's inferred from the `--output_file` extension.

The biggest difference right now is the treatment of stdin.
With pandoc you can convert stdin with

    ```
    $ cat input.md | pandoc
    ```

With `stitch`, it's written as

    ```
    $ cat input.md | stitch -
    ```

So a `-` is the marker for stdin.

## Notes on Pandoc

If you aren't familiar, pandoc is a universal document converter written
in Haskell.

For outputs like HTML and LaTeX, pandoc produces document fragments by default.
This means the converted output doesn't include things like `<head>`
tags, just the body.
`stitch`, on the other hand, prefers standalone documents.

The second difference is that stitch prefers self-contained documents.
This means things like images are inlined as base64 encoded PNGs or SVGs.
With pandoc, this options are enabled with the

- `--standalone`
- `--self-contained`

options. With `stitch`, the defaults are flipped to True, and disabled with

- `--no-standalone`
- `--no-self-contained`




[^literal_code]: If you look at the markdown source for this document,
you'll see that I've indented the code block by 4 spaces. This is so that
the block is interpreted as a literal chunk of text, and isn't intercepted
by the engine for execution. To actually run code the code chunk should be at
the start of the line.

