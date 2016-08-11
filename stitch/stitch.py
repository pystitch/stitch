"""
The short version is

1. Use pandoc / pypandoc to convert markdown (or whaterver)
to pandoc's JSON AST
2. Walk the AST looking for code-blocks that are to be executed
3. Submit code-blocks to the appropriate kernel
4. Capture output
5. Convert outupt to pandoc's JSON AST
6. stitch output in after the original code block
7. Use pandoc / pypandoc to convert the AST to output.

"""
import copy
import json
from collections.abc import MutableMapping
from queue import Empty
from collections import namedtuple
from jupyter_client.manager import start_new_kernel
from pandocfilters import Para, Str, RawBlock, Div
import pypandoc

# see https://github.com/jupyter/nbconvert/blob/master/nbconvert/preprocessors/execute.py
CODE = 'code'
CODEBLOCK = 'CodeBlock'
OUTPUT_FORMATS = ['html']

KernelPair = namedtuple("KernelPair", "km kc")


# --------
# User API
# --------

def convert_file(input_file, to, extra_args=(), outputfile=None,
                 filters=None):
    with open(input_file) as f:
        source = f.read()

    return convert(source, to, extra_args=extra_args, outputfile=outputfile,
                   filters=filters)

def convert(source: str, to: str, extra_args=(), outputfile=None,
            filters=None) -> None:
    """
    Convert a source document to an output file.
    """
    newdoc = stitch(source)
    return pypandoc.convert_text(newdoc, to, format='json',
                                 outputfile=outputfile)


def stitch(source: str, kernel_name='python') -> str:
    """
    Execute code blocks, stitching the outputs back into a file.
    """
    meta, blocks = tokenize(source)
    needed_kernels = set(extract_kernel_name(block) for block in blocks
                         if to_execute(block))
    kernels = {name: KernelPair(*start_new_kernel(kernel_name=name))
               for name in needed_kernels}
    for name, kernel in kernels.items():
        initialize_graphics(name, kernel)

    new_blocks = []
    for block in blocks:
        if is_code_block(block):
            new_blocks.append(wrap_input_code(block))
        else:
            new_blocks.append(block)
        if to_execute(block):
            result = execute_block(block, kernels)
            if result and result[0] is not None:
                result = wrap_output(result)
                new_blocks.append(result)

    doc = json.dumps([meta, new_blocks])
    return doc


def is_code_block(block):
    is_code = block['t'] == CODEBLOCK
    return is_code  # TODO: echo


def wrap_input_code(block):
    # TODO: IPython In / Out formatting
    new = copy.deepcopy(block)
    code = block['c'][1]
    new['c'][1] = format_input_code(code)
    return new

def format_input_code(code):
    return '## ' + code.replace('\n', '\n## ')


def parse_kernel_arguments(block):
    """
    Parse the kernel arguments of a code block,
    returning a tuple of (args, kwargs)

    Parameters
    ----------
    block

    Returns
    -------
    tuple

    Notes
    -----
    The allowed positional arguments are

    - kernel_name
    - chunk_name

    All other arguments must be like ``keyword=value``.
    """
    options = block['c'][0][1]
    kernel_name = chunk_name = None
    if len(options) == 0:
        pass
    elif len(options) == 1:
        kernel_name = options[0]
    elif len(options) == 2:
        kernel_name, chunk_name = options
    else:
        raise TypeError("Bad options %s" % options)
    kwargs = dict(block['c'][0][2])
    return (kernel_name, chunk_name), kwargs


def extract_kernel_name(block):
    options = block['c'][0][1]
    if len(options) >= 1:
        return options[0]
    else:
        return None


def wrap_output(output):
    out = output[-1]  # TODO: Multiple outputs
    if not isinstance(out, MutableMapping):
        out = {'text/plain': out}  # TODO, this is from print...
        output = [out]

    # TODO: this is getting messy
    order = ['text/plain', 'text/html', 'image/svg+xml', 'image/png']
    key = sorted(out, key=lambda x: order.index(x))[-1]
    if key == 'text/plain':
        # ident, classes, kvs
        return Div(['', ['output'], []], [Para([Str(output[-1][key])])])
    elif key in ('text/html', 'image/svg+xml'):
        return RawBlock('html', output[-1][key])
    elif key == 'image/png':
        data = '<img src="data:image/png;base64,{}">'.format(output[-1][key])
        return RawBlock('html', data)

    return Para([Str(output[-1][key])])


def tokenize(source: str) -> dict:
    return json.loads(pypandoc.convert_text(source, 'json', 'markdown'))


def to_execute(x):
    return (x['t'] == CODEBLOCK and ['eval', 'False'] not in x['c'][0][2] and
            extract_kernel_name(x) is not None)


def execute_block(block, kernels, timeout=None):
    # see nbconvert.run_cell
    kc = kernels[extract_kernel_name(block)]
    code = block['c'][1]
    outs = run_code(code, kc, timeout=timeout)
    return outs
    # log


def run_code(code, kp, timeout=None):
    msg_id = kp.kc.execute(code)
    while True:
        try:
            msg = kp.kc.shell_channel.get_msg(timeout=timeout)
        except Empty:
            pass
            # self.log.error(
            #     "Timeout waiting for execute reply (%is)." % self.timeout)
            # if self.interrupt_on_timeout:
            #     self.log.error("Interrupting kernel")
            #     self.km.interrupt_kernel()
            # else:
            #     try:
            #         exception = TimeoutError
            #     except NameError:
            #         exception = RuntimeError
            #     raise exception(
            #         "Cell execution timed out, see log for details.")

        if msg['parent_header'].get('msg_id') == msg_id:
            break
        else:
            # not our reply
            continue

    outs = []

    while True:
        try:
            # We've already waited for execute_reply, so all output
            # should already be waiting. However, on slow networks, like
            # in certain CI systems, waiting < 1 second might miss messages.
            # So long as the kernel sends a status:idle message when it
            # finishes, we won't actually have to wait this long, anyway.
            msg = kp.kc.iopub_channel.get_msg(timeout=4)
        except Empty:
            pass
            # self.log.warn("Timeout waiting for IOPub output")
            # if self.raise_on_iopub_timeout:
            #    # raise RuntimeError("Timeout waiting for IOPub output")
        if msg['parent_header'].get('msg_id') != msg_id:
            # not an output from our execution
            continue

        msg_type = msg['msg_type']
        # self.log.debug("output: %s", msg_type)
        content = msg['content']

        if msg_type == 'status':
            if content['execution_state'] == 'idle':
                break
            else:
                continue
        elif msg_type == 'execute_input':
            continue
        elif msg_type == 'clear_output':
            outs = []
            continue
        elif msg_type.startswith('comm'):
            continue

        try:
            out = output_from_msg(msg)
        except ValueError:
            pass
            # self.log.error("unhandled iopub msg: " + msg_type)
        else:
            outs.append(out)

    return outs


def output_from_msg(msg):
    """

    """
    content = msg['content']
    return content.get('data') or content.get('text')


def initialize_graphics(name, kp):
    # TODO: set_matplotlib_formats takes *args
    # TODO: do as needed? Push on user?
    # valid_formats = ["png", "jpg", "jpeg", "pdf", "svg"]
    if name == 'python':
        code = """\
        %matplotlib inline
        from IPython.display import set_matplotlib_formats
        """
        kp.kc.execute(code + 'set_matplotlib_formats("png")',
                      store_history=False)
        # fmt_code = '\n'.join("set_matplotlib_formats('{}')".format(fmt)
        #                      for fmt in valid_formats)
        # code = dedent(code) + fmt_code
        # kp.kc.execute(code, store_history=False)
    else:
        raise ValueError(name)



