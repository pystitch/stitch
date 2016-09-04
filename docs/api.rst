API
===

.. autoclass:: stitch.Stitch
   :members:

Chunk Options
~~~~~~~~~~~~~

Code chunks are blocks that look like

.. code-block:: none

   ```{kernel_name, [chunk_name], **kwargs}
   # code
   ```

The ``kernel_name`` is required, and ``chunk_name`` is optional.
All parameters are separated by a comma.

   .. method:: kernel_name(name: str)

      Name of the kernel to use for executing the code chunk.
      Required. See ``jupyter kernelspec list``.

   .. method:: chunk_name(chunk_name: str)

      Name for the chunk. Controls the filename for supporting files
      created by that chunk. Optional.

   .. method:: echo(echo=True)

      whether to include the input-code in the rendered output.
      Default True.

   .. method:: eval(eval=True)

      Whether to execute the code cell. Default True.

   .. method:: results(s)

      str; how to display the results

      * hide: hide the chunk output (but still execute the chunk)

   .. method:: warning(True)

      bool; whether to include warnings (stderr) in the ouput.

   .. method:: width(w)

      Width for output figure. See http://pandoc.org/MANUAL.html#images

      .. warning::

         This will probably change to ``fig.width`` in a future release.

   .. method:: height(w)

      Height for output figure. See http://pandoc.org/MANUAL.html#images

      .. warning::

         This will probably change to ``fig.height`` in a future release.

