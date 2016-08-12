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
import os
import copy
import json
from typing import List, Tuple, Optional
from collections.abc import MutableMapping
from queue import Empty
from collections import namedtuple
from jupyter_client.manager import start_new_kernel
from pandocfilters import Para, Str, RawBlock, Div
import pypandoc

# see https://github.com/jupyter/nbconvert/blob/master/nbconvert/preprocessors/execute.py
CODE = 'code'
CODEBLOCK = 'CodeBlock'
OUTPUT_FORMATS = ['html', 'latex']
HERE = os.path.dirname(__file__)
CSS = os.path.join(HERE, 'static', 'default.css')

KernelPair = namedtuple("KernelPair", "km kc")


# --------
# User API
# --------

def convert_file(input_file, to, extra_args=None, output_file=None,
                 filters=None):
    """
    Convert a markdown ``input_file`` to ``to``.

    Parameters
    ----------
    input_file : str
    to : str
    extra_args : iterable
    output_file : str
    filters : iterable
    """
    with open(input_file) as f:
        source = f.read()
    if extra_args is None:
        extra_args = ['--standalone', '--css=%s' % CSS]

    return convert(source, to, extra_args=extra_args, output_file=output_file,
                   filters=filters)


def convert(source: str, to: str, extra_args=(), output_file=None,
            filters=None) -> None:
    """
    Convert a source document to an output file.
    """
    newdoc = stitch(source)
    result = pypandoc.convert_text(newdoc, to, format='json',
                                   extra_args=extra_args,
                                   outputfile=output_file)
    if output_file is None:
        print(result)


def kernel_factory(kernel_name):
    print('Starting kernel: ', kernel_name)
    return KernelPair(*start_new_kernel(kernel_name=kernel_name))


def stitch(source: str) -> str:
    """
    Execute code blocks, stitching the outputs back into a file.
    """
    meta, blocks = tokenize(source)
    kernels = {}

    for name, kernel in kernels.items():
        initialize_graphics(name, kernel)

    new_blocks = []
    for block in blocks:
        if not is_code_block(block):
            new_blocks.append(block)
            continue
        # We should only have code blocks now...
        # Execute first, to get prompt numbers
        (lang, name), attrs = parse_kernel_arguments(block)
        if is_executable(block, lang, attrs):
            # still need to check, since kernel_factory(lang) is executaed
            # even if the key is present
            if lang not in kernels:
                kernel = kernels.setdefault(lang, kernel_factory(lang))
            message_pairs = execute_block(block, kernel)
            execution_count = extract_execution_count(message_pairs)

        # Now handle input formatting
        if attrs.get('echo', True):
            new_blocks.append(wrap_input_code(block, execution_count))

        # And output formatting
        if is_stitchable(message_pairs, attrs):
            result = wrap_output(message_pairs, execution_count)
            new_blocks.append(result)

    doc = json.dumps([meta, new_blocks])
    return doc

# -----------
# Input Tests
# -----------

def is_code_block(block):
    is_code = block['t'] == CODEBLOCK
    return is_code  # TODO: echo


def is_executable(x, lang, attrs):
    return (x['t'] == CODEBLOCK and attrs.get('eval') is not False and
            lang is not None)


# ------------
# Output Tests
# ------------

def is_stitchable(result, attrs):
    return (bool(result) and
            result[0] is not None and
            attrs.get('results') != 'hide')


# ----------
# Formatting
# ----------
def format_input_prompt(code, number):
    start = 'In [{}]: '.format(number)
    split = code.split('\n')
    rest = ['{}...: '.format(' ' * (len(start) - 5)) for x in split[1:]]
    formatted = '\n'.join(l + r for l, r in zip([start] + rest, split))
    return formatted


def format_output_prompt(output, number):
    pass


# ----------------
# Input Processing
# ----------------

def tokenize(source: str) -> dict:
    return json.loads(pypandoc.convert_text(source, 'json', 'markdown'))


def wrap_input_code(block, execution_count):
    # TODO: IPython In / Out formatting
    new = copy.deepcopy(block)
    code = block['c'][1]
    new['c'][1] = format_input_prompt(code, execution_count)
    return new


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
        kernel_name = options[0].strip('{}').strip()
    elif len(options) == 2:
        kernel_name, chunk_name = options
    else:
        raise TypeError("Bad options %s" % options)
    kwargs = dict(block['c'][0][2])
    kwargs = {k: v == 'True' if v in ('True', 'False') else v
              for k, v in kwargs.items()}

    return (kernel_name, chunk_name), kwargs


def extract_kernel_name(block):
    options = block['c'][0][1]
    if len(options) >= 1:
        return options[0].strip('{}').strip()
    else:
        return None


# -----------------
# Output Processing
# -----------------


def wrap_output(message_pairs, execution_count):
    output = [message[0] for message in message_pairs]
    out = output[-1]  # TODO: Multiple outputs
    if not isinstance(out, MutableMapping):
        out = {'text/plain': out}  # TODO, this is from print...
        output = [out]

    # TODO: this is getting messy
    order = ['text/plain', 'text/latex', 'text/html', 'image/svg+xml', 'image/png']
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


# --------------
# Code Execution
# --------------

def execute_block(block, kp, timeout=None):
    # see nbconvert.run_cell
    code = block['c'][1]
    message_pairs = run_code(code, kp, timeout=timeout)
    return message_pairs


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

    message_pairs = []

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
            message_pairs = []
            continue
        elif msg_type.startswith('comm'):
            continue

        try:
            out = output_from_msg(msg)
        except ValueError:
            pass
            # self.log.error("unhandled iopub msg: " + msg_type)
        else:
            message_pairs.append((out, msg))

    return message_pairs


def output_from_msg(msg):
    """

    """
    content = msg['content']
    return content.get('data') or content.get('text')


def extract_execution_count(
        message_pairs: List[Tuple[dict, dict]]) -> Optional[int]:
    '''
    '''
    messages = [x[1] for x in message_pairs]
    for message in messages:
        count = message['content'].get('execution_count')
        if count is not None:
            return count


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
        # raise ValueError(name)
        pass
