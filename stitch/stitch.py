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
# Adapted from knitpy and nbcovert:
# Copyright (c) Jan Schulz <jasc@gmx.net>
# Copyright (c) IPython Development Team.
# Distributed under the terms of the Modified BSD License.
import os
import copy
import json
from typing import List, Optional
from collections import namedtuple
from queue import Empty

from jupyter_client.manager import start_new_kernel
from nbconvert.utils.base import NbConvertBase
from pandocfilters import Para, Str, RawBlock, Div
import pypandoc

DISPLAY_PRIORITY = NbConvertBase().display_data_priority
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
            messages = execute_block(block, kernel)
            execution_count = extract_execution_count(messages)

        # Now handle input formatting
        if attrs.get('echo', True):
            new_blocks.append(wrap_input_code(block, execution_count))

        # And output formatting
        if is_stitchable(messages, attrs):
            result = wrap_output(messages, execution_count)
            new_blocks.extend(result)

    doc = json.dumps([meta, new_blocks])
    return doc

# -----------
# Input Tests
# -----------

def is_code_block(block):
    is_code = block['t'] == CODEBLOCK
    return is_code  # TODO: echo


def is_executable(x, lang, attrs):
    return (is_code_block(x) and attrs.get('eval') is not False and
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


def wrap_input_code(block, execution_count):
    new = copy.deepcopy(block)
    code = block['c'][1]
    new['c'][1] = format_input_prompt(code, execution_count)
    return new


def format_output_prompt(output, number):
    pass


# ----------------
# Input Processing
# ----------------

def tokenize(source: str) -> dict:
    return json.loads(pypandoc.convert_text(source, 'json', 'markdown'))


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

def plain_output(text):
    block = Div(['', ['output'], []], [Para([Str(text)])])
    return block


def pytb(text):
    # TODO
    pass


def wrap_output(messages, execution_count):
    '''
    stdout is wrapped in a code block?
    other stuff is wrapped.

    return a list of blocks
    '''
    # messsage_pairs can come from stdout or the io stream (maybe others?)
    std_out_messages = [x for x in messages if is_stdout(x)]
    display_messages = [x for x in messages if not is_stdout(x)]

    output_blocks = []

    # Handle all stdout first...
    for message in std_out_messages:
        text = message['content']['text']
        output_blocks.append(plain_output(text))

    order = dict(
        (x[1], x[0]) for x in enumerate(NbConvertBase().display_data_priority)
    )

    for message in display_messages:
        if message['header']['msg_type'] == 'error':
            block = plain_output('\n'.join(message['content']['traceback']))
        else:
            # TODO: traceback
            all_data = message['content']['data']
            key = sorted(all_data.keys(), key=lambda k: order[k])[0]
            data = all_data[key]

            if key == 'text/plain':
                # ident, classes, kvs
                block = plain_output(data)
            elif key in ('text/html', 'image/svg+xml'):
                block = RawBlock('html', data)
            elif key == 'image/png':
                block = RawBlock(
                    'html', '<img src="data:image/png;base64,{}">'.format(data)
                )
        output_blocks.append(block)
    return output_blocks


def is_stdout(message):
    return message['content'].get('name') == 'stdout'

# --------------
# Code Execution
# --------------
def execute_block(block, kp, timeout=None):
    # see nbconvert.run_cell
    code = block['c'][1]
    messages = run_code(code, kp, timeout=timeout)
    return messages


def run_code(code: str, kp: KernelPair, timeout=None) -> List:
    '''
    Execute a code chunk, capturing the output.

    Parameters
    ----------
    code : str
    kp : KernelPair

    Returns
    -------
    outputs : List
    '''
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

    messages = []

    while True:  # until idle message
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
            messages = []
            continue
        elif msg_type.startswith('comm'):
            continue

        try:
            messages.append(msg)
        except ValueError:
            pass
            # self.log.error("unhandled iopub msg: " + msg_type)

    return messages


def extract_execution_count(
        messages: List[dict]) -> Optional[int]:
    '''
    '''
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
