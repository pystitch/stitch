# Stitch

[stitch](https://github.com/TomAugspurger/stitch) is a library for
making reproducible reports.
It takes a markdown source file, executes the code chunks, captures the output,
and stitches the output into the destination file.

You should consider using [knitpy](https://github.com/janschulz/knitpy/) instead, or [rmarkdown](http://rmarkdown.rstudio.com) if you only need R.

While `stitch` is written in python, in principle it can be used for any
of the dozens of [Jupyter kernels](https://github.com/ipython/ipython/wiki/IPython-kernels-for-other-languages).

## Why Not Jupyter Notebooks?

Use both! Notebooks are great for interactive computing, and you should
use them whenever appropriate.
You can do some great things with a notebook and `nbconvert`.

That said, they're not my favorite environment for writing long-form text,
with bits of code mixed in.
I'd much rather be in my favorite text-editor.
To over-simplify a bit, any document has a ratio of text : code.
When that ratio skews towards the code end, I prefer the notebook.
When it's skewed to the text end, I prefer a markdown file

## What's in a Name?

The name `stitch` has a couple meanings. Like R's `knit`, we are taking a source
document, executing code chunks, and `knit`ing or `stitch`ing the output
back into the document.

The second meaning is for `stitch` bringing together a bunch of great libraries,
minimizing the work we have to do ourselves. `stitch` uses

- [Pandoc](http://pandoc.org/MANUAL.html) markdown parsing and conversion to the destination output
- [jupyter](http://jupyter.org), specifically [jupyter-client](https://jupyter-client.readthedocs.io/en/latest/) for managing kernels, passing code to kernels,
and capturing the output
- [pandocfilters](https://github.com/jgm/pandocfilters/) for converting code-chunk
output to pandoc's AST
- [pypandoc](https://github.com/coldfix/pypandoc) for communicating with pandoc
- [click](http://click.pocoo.org) for the command-line interface.

`stitch` itself is fairly minimal.
The main tasks are

- processing code-chunk arguments
- passing the correct outputs from the jupyter kernel to pandocfilters
- assembling all the chunks of text, code, and output in a sensible way
- making things look nice
