Stitch
======

|Build Status|

A `knitr <http://yihui.name/knitr/>`__- `RMarkdown <http://rmarkdown.rstudio.com>`__-like library, in Python.

*Note:* You might want to consider Jan Schulz's `knitpy <https://github.com/janschulz/knitpy/>`__
instead. It's probably more mature at this point. However, I wanted
to see if there was a simpler way of doing things.

The high-level goal of this type of library (knitr/RMarkdown, knitpy, and stitch) is to make writing
reproducible reports easier.

Documentation is available `here <https://pystitch.github.io>`__.

Examples
========

See the project's `examples page <https://pystitch.github.io/_downloads/side_by_side.html>`__ for a
side-by-side comparison of input markdown and stitched HTML.

More complex examples are linked to from there as well.

Install
=======

``stitch`` supports Python 3.5 and above.
At the moment ``stitch`` can be installed from pip via

.. code-block:: sh

   pip install knotr

I know, it's confusing.
I've filed a claim for ``stitch`` on PyPI, but I think the people working that support queue are over-worked.
Once that gets processed, I'll put it up on conda-forge as well.
If you need a mnemonic, it's "I want knitr, but `not` the one written in `R`."
Also I wanted to confuse R users.
And knots are kind of like a buggy version of knits.

``stitch`` requires pandoc>=1.18. This can be installed using your
system package manager, or `pypandoc <https://pypi.python.org/pypi/pypandoc>`__.

Design
======

The goal was to keep ``stitch`` itself extremely simple by reusing
existing libraries. A high level overview of our tasks is

1. Command-line Interface
2. Parse markdown file
3. Execute code chunks, capturing the output
4. Collate execution output into the document
5. Render to final output

Fortunately the building blocks are all there.

We reuse

-  `pandoc <http://pandoc.org>`__ via
   `pypandoc <https://pypi.python.org/pypi/pypandoc>`__ for parsing
   markdown and rendering the final output
-  `jupyter <http://jupyter.readthedocs.io/en/latest/>`__ for
   language kernels, executing code, and collecting the output
-  Use `pandocfilters <https://github.com/jgm/pandocfilters>`__ to
   collate the execution output into the document

So all ``stitch`` has to do is to provide a command-line interface, scan
the document for code chunks, manage some kernels, hand the code to the
kernels, pass the output to an appropriate ``pandocfilter``.

The biggest departure from ``knitpy`` is the use of pandoc's JSON AST.
This is what you get from ``pandoc -t json input.md``

This saves us from having do any kind of custom parsing of the markdown.
The only drawback so far is somewhat inscrutable Haskell exceptions if
``stitch`` happens to produce a bad document.

Documentation
=============

Stitch's documentation has an odd build process, so standard tools like
readthedocs weren't flexible enough.
To make the docs, install stitch and all the extra dependencies.
Clone https://github.com/pystitch/pystitch.github.io

Checkout the `src` branch.

Run `make html`.

.. |Build Status| image:: https://travis-ci.org/pystitch/stitch.svg?branch=master
   :target: https://travis-ci.org/pystitch/stitch
