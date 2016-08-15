"""
Convert markdown files, executing code chunks and stitching
in the output.
"""
# Adapted from knitpy and nbcovert:
# Copyright (c) Jan Schulz <jasc@gmx.net>
# Copyright (c) IPython Development Team.
# Distributed under the terms of the Modified BSD License.
import os
import re
import copy
import json
from typing import List, Optional, Iterable
from collections import namedtuple
from queue import Empty

from jupyter_client.manager import start_new_kernel
from nbconvert.utils.base import NbConvertBase
from pandocfilters import Para, Str, RawBlock, Div
import pypandoc

DISPLAY_PRIORITY = NbConvertBase().display_data_priority
CODE = 'code'
CODEBLOCK = 'CodeBlock'
OUTPUT_FORMATS = ['html', 'latex']
HERE = os.path.dirname(__file__)
CSS = os.path.join(HERE, 'static', 'default.css')

KernelPair = namedtuple("KernelPair", "km kc")
CODE_CHUNK_XPR = re.compile(r'^```{\w+.*}')


# --------
# User API
# --------

def convert_file(input_file: str,
                 to: str,
                 extra_args: Iterable[str]=None,
                 output_file: Optional[str]=None) -> None:
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
    # TODO: Put all settings app configurable.
    if extra_args is None:
        extra_args = ['--standalone', '--css=%s' % CSS]

    convert(source, to, extra_args=extra_args, output_file=output_file)


def convert(source: str, to: str, extra_args: Iterable[str]=(),
            output_file: str=None) -> None:
    """
    Convert a source document to an output file.
    """
    newdoc = stitch(source)
    result = pypandoc.convert_text(newdoc, to, format='json',
                                   extra_args=extra_args,
                                   outputfile=output_file)
    if output_file is None:
        print(result)


def kernel_factory(kernel_name: str) -> KernelPair:
    """
    Start a new kernel.

    Parameters
    ----------
    kernel_name : str

    Returns
    -------
    KernalPair: namedtuple
      - km (KernelManager)
      - kc (KernelClient)
    """
    return KernelPair(*start_new_kernel(kernel_name=kernel_name))


def stitch(source: str) -> str:
    """
    Execute code blocks, stitching the outputs back into a file.
    """
    source = preprocess(source)
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
            # even if the key is present, only want one kernel / lang
            if lang not in kernels:
                kernel = kernels.setdefault(lang, kernel_factory(lang))
            messages = execute_block(block, kernel)
            execution_count = extract_execution_count(messages)

        # ... now handle input formatting...
        if attrs.get('echo', True):
            new_blocks.append(wrap_input_code(block, execution_count))

        # ... and output formatting
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


def is_executable(block, lang, attrs):
    """
    Return whether a block should be executed.
    Must be a code_block, and must not have ``eval=False`` in the block
    arguments, and ``lang`` (kernel_name) must not be None.
    """
    return (is_code_block(block) and attrs.get('eval') is not False and
            lang is not None)


# ------------
# Output Tests
# ------------

def is_stitchable(result, attrs):
    """
    Return whether an output ``result`` should be included in the output.
    ``result`` should not be empty or None, and ``attrs`` should not
    include ``{'results': 'hide'}``.
    """
    return (bool(result) and
            result[0] is not None and
            attrs.get('results') != 'hide')


# ----------
# Formatting
# ----------
def format_input_prompt(code, number):
    """
    Wrap the input code in IPython style ``In [X]:`` markers.
    """
    start = 'In [{}]: '.format(number)
    split = code.split('\n')

    def trailing_space(x):
        # all blank lines shouldn't have a trailing space after ...:
        return '' if x == '' else ' '

    rest = ['{}...:{}'.format(' ' * (len(start) - 5),
                              trailing_space(x))
            for x in split[1:]]
    formatted = '\n'.join(l + r for l, r in zip([start] + rest, split))
    return formatted


def wrap_input_code(block, execution_count):
    new = copy.deepcopy(block)
    code = block['c'][1]
    new['c'][1] = format_input_prompt(code, execution_count)
    return new


def format_output_prompt(output, number):
    # TODO
    pass


# ----------------
# Input Processing
# ----------------

def tokenize(source: str) -> dict:
    """
    Convert a document to pandoc's JSON AST.
    """
    return json.loads(pypandoc.convert_text(source, 'json', 'markdown'))


def _transform(kind, text):
    if kind == 'ARG':
        result = '.' + text
    elif kind in ('DELIM' 'BLANK'):
        result = None
    elif kind in ('OPEN', 'CLOSE', 'KWARG'):
        return text
    else:
        raise TypeError('Unknown kind %s' % kind)
    return result


def is_code_chunk_line(line):
    return CODE_CHUNK_XPR.match(line)


def validate_options(options_line):
    xpr = re.compile(r'^```{\w+.*}')
    if not xpr.match(options_line):
        raise TypeError("Invalid chunk options %s" % options_line)


def preprocess(source):
    doc = []
    for line in source.split('\n'):
        if CODE_CHUNK_XPR.match(line):
            doc.append(preprocess_options(line))
        else:
            doc.append(line)
    return '\n'.join(doc)


def preprocess_options(options_line):
    # See Python Cookbook 3rd Ed p 67
    KWARG = r'(?P<KWARG>\w+ *= *\w+)'
    ARG = r'(?P<ARG>\w+)'
    DELIM = r'(?P<DELIM> *, *)'
    BLANK = r'(?P<BLANK>\s+)'
    OPEN = r'(?P<OPEN>```{ *)'
    CLOSE = r'(?P<CLOSE>})'

    Token = namedtuple("Token", ['kind', 'value'])
    master_pat = re.compile('|'.join([KWARG, ARG, DELIM, OPEN, CLOSE, BLANK]))

    def generate_tokens(pat, text):
        scanner = pat.scanner(text)
        for m in iter(scanner.match, None):
            yield Token(m.lastgroup, m.group())

    items = (_transform(kind, text) for kind, text in generate_tokens(master_pat, options_line))
    items = filter(None, items)
    items = ' '.join(items)
    result = items.replace('{ ', '{').replace(' }', '}')
    return result


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
    output_messages = [x for x in messages if not is_execute_input(x)]
    std_out_messages = [x for x in output_messages if is_stdout(x)]
    display_messages = [x for x in output_messages if not is_stdout(x)]

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
            key = min(all_data.keys(), key=lambda k: order[k])
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


def is_execute_input(message):
    return message['msg_type'] == 'execute_input'


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

    Notes
    -----
    See https://github.com/jupyter/nbconvert/blob/master/nbconvert
      /preprocessors/execute.py
    '''
    msg_id = kp.kc.execute(code)
    while True:
        try:
            msg = kp.kc.shell_channel.get_msg(timeout=timeout)
        except Empty:
            # TODO: Log error
            pass

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
            # TODO: Log error
        if msg['parent_header'].get('msg_id') != msg_id:
            # not an output from our execution
            continue

        msg_type = msg['msg_type']
        content = msg['content']

        if msg_type == 'status':
            if content['execution_state'] == 'idle':
                break
            else:
                continue
        elif msg_type in ('execute_input', 'execute_result', 'display_data',
                          'stream'):
            # Keep `execute_input` just for execution_count if there's
            # no result
            messages.append(msg)
        elif msg_type == 'clear_output':
            messages = []
            continue
        elif msg_type.startswith('comm'):
            continue
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
