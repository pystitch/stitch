# Stitch

[![Build Status](https://travis-ci.org/TomAugspurger/stitch.svg?branch=master)](https://travis-ci.org/TomAugspurger/stitch)

An experimental knitr-like library, in Python.

You should use [knitpy](https://github.com/janschulz/knitpy/) instead.
This is an unfinished, worse version of it.
However, I wanted to see if there was a simpler way of doing things.

# Design

The goal was to keep `stitch` itself extremely simple by reusing existing libraries.
A high level overview of our tasks is

1. Command-line Interface
2. Parse markdown file
3. Execute code chunks, capturing the output
4. Collate execution output into the document
5. Render to final output

Fortunately the building blocks are all there.

We reuse

- [`pandoc`](http://pandoc.org) via [`pypandoc`](https://pypi.python.org/pypi/pypandoc) for parsing markdown and rendering the final output
- [`jupyter`](http://jupyter.readthedocs.io/en/latest/) for language kernels, executing code, and collecting the output
- Use [`pandocfilters`](https://github.com/jgm/pandocfilters) to collate the execution output into the document

So all `stitch` has to do is to provide a command-line interface, scan the document for code chunks, manage some kernels, hand the code to the kernels, pass the output to an appropriate `pandocfilter`.

The biggest departure from `knitpy` is the use of pandoc's JSON AST.
This is what you get from `pandoc -t json input.md`

This saves us from having do any kind of custom parsing of the markdown.
The only drawback so far is somewhat inscrutable Haskell exceptions if `stitch`
happens to produce a bad document.

# An Example

Code chunks are specified using backticks

```{.python}
%matplotlib inline
import pandas as pd
import seaborn as sns
df = sns.load_dataset("iris")
```

A kernel maintains the state between code-chunks.
Graphics are included.
We can take an example from [seaborn](https://stanford.edu/~mwaskom/software/seaborn/examples/scatterplot_matrix.html):

```{.python}
sns.set()
sns.pairplot(df, hue="species");
```

![Side-by-side](comparison.png)

