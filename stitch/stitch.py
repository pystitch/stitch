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
import base64
from typing import List, Optional, Iterable
from collections import namedtuple
from queue import Empty

from jupyter_client.manager import start_new_kernel
from nbconvert.utils.base import NbConvertBase
from pandocfilters import RawBlock, Div, CodeBlock, Image, Str, Para
import pypandoc

from .exc import StitchError

DISPLAY_PRIORITY = NbConvertBase().display_data_priority
CODE = 'code'
CODEBLOCK = 'CodeBlock'
OUTPUT_FORMATS = ['html', 'latex']
HERE = os.path.dirname(__file__)

KernelPair = namedtuple("KernelPair", "km kc")
CODE_CHUNK_XPR = re.compile(r'^```{\w+.*}|^```\w+')


# --------
# User API
# --------

class Stitch:
    '''
    Helper class for managing the execution of a document.
    Stores configuration variables.
    '''

    def __init__(self, name, to='html', standalone=True,
                 on_error='continue'):
        '''
        Parameters
        ----------
        name : str
            controls the directory for supporting files
        to : str, default ``'html'``
            output format
        standalone : bool, default True
            whether to make a standalone document
        on_error : ``{"continue", "raise"}``
            how to handle errors in the code being executed.
        '''
        self.name = name
        self.to = to
        self.standalone = standalone
        self._kernel_pairs = {}

        self.resource_dir = self.name_resource_dir(name)
        self.on_error = on_error

    @staticmethod
    def name_resource_dir(name):
        '''
        Give the directory name for supporting resources
        '''
        return '{}_files'.format(name)

    @property
    def self_contained(self):
        # return self.to not in ('pdf', 'docx')
        return True

    @property
    def kernel_managers(self):
        '''
        dict of KernelManager, KernelClient pairs, keyed by
        kernel name.
        '''
        return self._kernel_pairs

    @property
    def on_error(self):
        '''
        How to handle errors in the code being executed. Must be one of
        ``{'continue', 'raise'}``.
        '''
        return self._on_error

    @on_error.setter
    def on_error(self, on_error):
        valid = {'continue', 'raise'}
        if on_error not in valid:
            msg = "`on_error` must be one of %s, got %s instead" % (valid,
                                                                    on_error)
            raise TypeError(msg)
        self._on_error = on_error

    def get_kernel(self, kernel_name):
        '''
        Get a kernel from ``kernel_managers`` by ``kernel_name``,
        creating it if needed.

        Parameters
        ----------
        kernel_name : str

        Returns
        -------
        kp : KernelPair
        '''
        kp = self.kernel_managers.get(kernel_name)
        if not kp:
            kp = kernel_factory(kernel_name)
            initialize_kernel(kernel_name, kp)
            self.kernel_managers[kernel_name] = kp
        return kp

    def stitch(self, source):
        '''
        Main method for converting a document.

        Parameters
        ----------
        source : str
            the actual text to be converted

        Returns
        -------
        meta, blocks : list
            These should be compatible with pando's JSON AST
        '''
        source = preprocess(source)
        meta, blocks = tokenize(source)
        new_blocks = []

        for i, block in enumerate(blocks):
            if not is_code_block(block):
                new_blocks.append(block)
                continue
            # We should only have code blocks now...
            # Execute first, to get prompt numbers
            (lang, name), attrs = parse_kernel_arguments(block)
            if name is None:
                name = "unnamed_chunk_{}".format(i)
            if is_executable(block, lang, attrs):
                # still need to check, since kernel_factory(lang) is executaed
                # even if the key is present, only want one kernel / lang
                kernel = self.get_kernel(lang)
                messages = execute_block(block, kernel)
                execution_count = extract_execution_count(messages)
            else:
                execution_count = None
                messages = []

            # ... now handle input formatting...
            if attrs.get('echo', True):
                new_blocks.append(wrap_input_code(block, execution_count))

            # ... and output formatting
            if is_stitchable(messages, attrs):
                result = self.wrap_output(
                    name, messages, execution_count, self.to, attrs,
                )
                new_blocks.extend(result)
        return meta, new_blocks

    def wrap_output(self, chunk_name, messages, execution_count, kp, attrs):
        '''
        Wrap the messages of a code-block.

        Parameters
        ----------
        chunk_name : str
        messages : list of dicts
        execution_count : int or None
        kp : KernelPair
        attrs : dict
            options from the source options-line.

        Returns
        -------
        output_blocks : list

        Notes
        -----
        Messages printed to stdout are wrapped in a CodeBlock.
        Messages publishing mimetypes (e.g. matplotlib figures)
        resuse Jupyter's display priority. See
        ``NbConvertBase.display_data_priority``.

        The result should be pandoc JSON AST compatible.
        '''
        # messsage_pairs can come from stdout or the io stream (maybe others?)
        output_messages = [x for x in messages if not is_execute_input(x)]
        std_out_messages = [x for x in output_messages if is_stdout(x)]
        std_err_messages = [x for x in output_messages if is_stderr(x)]
        display_messages = [x for x in output_messages if not is_stdout(x) and
                            not is_stderr(x)]

        output_blocks = []

        # Handle all stdout first...
        for message in std_out_messages + std_err_messages:
            text = message['content']['text']
            output_blocks.append(plain_output(text))

        order = dict(
            (x[1], x[0]) for x in enumerate(NbConvertBase().display_data_priority)
        )

        for message in display_messages:
            if message['header']['msg_type'] == 'error':
                if self.on_error == 'raise':
                    exc = StitchError(message['content']['traceback'])
                    raise exc
                block = plain_output('\n'.join(message['content']['traceback']))
            else:
                all_data = message['content']['data']
                key = min(all_data.keys(), key=lambda k: order[k])
                data = all_data[key]

                if self.to in ('latex', 'pdf', 'beamer'):
                    if 'text/latex' in all_data.keys():
                        key = 'text/latex'
                        data = all_data[key]

                if key == 'text/plain':
                    # ident, classes, kvs
                    block = plain_output(data)
                elif key == 'text/latex':
                    block = RawBlock('latex', data)
                elif key == 'text/html':
                    block = RawBlock('html', data)
                elif key.startswith('image'):
                    block = self.wrap_image_output(chunk_name, data, key, attrs)

            output_blocks.append(block)
        return output_blocks

    def wrap_image_output(self, chunk_name, data, key, attrs):
        '''
        Extra handling for images

        Parameters
        ----------
        chunk_name, data, key : str
        attrs: dict

        Returns
        -------
        Para[Image]
        '''
        # TODO: interaction of output type and standalone.
        # TODO: this can be simplified, do the file-writing in one step
        def b64_encode(data):
            return base64.encodestring(data.encode('ascii')).decode('ascii')

        # TODO: dict of attrs on Stitcher.
        image_attrs = {'width', 'height'}
        attrs = [(k, v) for k, v in attrs.items() if k in image_attrs]
        if self.self_contained:
            if 'png' in key:
                data = 'data:image/png;base64,{}'.format(data)
            elif 'svg' in key:
                data = 'data:image/svg+xml;base64,{}'.format(b64_encode(data))
            if 'png' in key or 'svg' in key:
                block = Para([Image([chunk_name, [], attrs],
                                    [Str("")],
                                    [data, ""])])
            else:
                raise TypeError("Unknown mimetype %s" % key)
        else:
            # we are saving to filesystem
            filepath = os.path.join(self.resource_dir,
                                    "{}.png".format(chunk_name))
            with open(filepath, 'wb') as f:
                f.write(base64.decodestring(data.encode('ascii')))
            # Image :: alt text (list of inlines), target
            # Image :: Attr [Inline] Target
            # Target :: (string, string)  of (URL, title)
            block = Para([Image([chunk_name, [], []],
                                [Str("")],
                                [filepath, "fig: {}".format(chunk_name)])])

        return block


def convert_file(input_file: str,
                 to: str,
                 extra_args: Iterable[str]=(),
                 output_file: Optional[str]=None) -> None:
    """
    Convert a markdown ``input_file`` to ``to``.

    Parameters
    ----------
    input_file : str
    to : str
    extra_args : iterable
    output_file : str

    See Also
    --------
    convert
    """
    with open(input_file) as f:
        source = f.read()
    convert(source, to, extra_args=extra_args, output_file=output_file)


def convert(source: str, to: str, extra_args: Iterable[str]=(),
            output_file: str=None) -> None:
    """
    Convert a source document to an output file.

    Parameters
    ----------
    source : str
    to : str
    extra_args : iterable
    output_file : str

    Notes
    -----
    Either writes to ``output_file`` or prints to stdout.
    """
    output_name = (
        os.path.splitext(os.path.basename(output_file))[0]
        if output_file is not None
        else 'std_out'
    )

    standalone = '--standalone' in extra_args
    stitcher = Stitch(name=output_name, to=to, standalone=standalone)
    meta, blocks = stitcher.stitch(source)
    result = json.dumps([meta, blocks])
    newdoc = pypandoc.convert_text(result, to, format='json',
                                   extra_args=extra_args,
                                   outputfile=output_file)

    if output_file is None:
        print(newdoc)


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


# -----------
# Input Tests
# -----------

def is_code_block(block):
    is_code = block['t'] == CODEBLOCK
    return is_code


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
    if number is None:
        return code
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


def validate_options(options_line):
    xpr = re.compile(r'^```{\w+.*}')
    if not xpr.match(options_line):
        raise TypeError("Invalid chunk options %s" % options_line)


def preprocess(source: str) -> str:
    """
    Process a source file prior to tokenezation.

    Parameters
    ----------
    source : str

    Returns
    -------
    processed : str

    Notes
    -----
    Currently applies the following transformations

    - preprocess_options: transform code chunk arguments
      to allow ``{python, arg, kwarg=val}`` instead of pandoc-style
      ``{.python .arg kwarg=val}``

    See Also
    --------
    prerpocess_options
    """
    doc = []
    for line in source.split('\n'):
        if CODE_CHUNK_XPR.match(line):
            doc.append(preprocess_options(line))
        else:
            doc.append(line)
    return '\n'.join(doc)


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


def preprocess_options(options_line):
    """
    Transform a code-chunk options line to allow
    ``{python, arg, kwarg=val}`` instead of pandoc-style
    ``{.python .arg kwarg=val}``.
    """
    # See Python Cookbook 3rd Ed p 67
    KWARG = r'(?P<KWARG>\w+ *= *\w+)'
    ARG = r'(?P<ARG>\w+)'
    DELIM = r'(?P<DELIM> *, *)'
    BLANK = r'(?P<BLANK>\s+)'
    OPEN = r'(?P<OPEN>```{ *)'
    CLOSE = r'(?P<CLOSE>})'

    Token = namedtuple("Token", ['kind', 'value'])
    master_pat = re.compile('|'.join([KWARG, ARG, DELIM, OPEN,
                                      CLOSE, BLANK]))

    def generate_tokens(pat, text):
        scanner = pat.scanner(text)
        for m in iter(scanner.match, None):
            yield Token(m.lastgroup, m.group())

    tok = list(generate_tokens(master_pat, options_line))

    items = (_transform(kind, text) for kind, text in tok)
    items = filter(None, items)
    items = ' '.join(items)
    result = items.replace('{ ', '{').replace(' }', '}').replace(" {", "{")
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
    block = Div(['', ['output'], []], [CodeBlock(['', [], []], text)])
    return block


def is_stdout(message):
    return message['content'].get('name') == 'stdout'


def is_stderr(message):
    return message['content'].get('name') == 'stderr'


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
                          'stream', 'error'):
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


def initialize_kernel(name, kp):
    # TODO: set_matplotlib_formats takes *args
    # TODO: do as needed? Push on user?
    # valid_formats = ["png", "jpg", "jpeg", "pdf", "svg"]
    if name == 'python':
        code = """\
        %colors NoColor
        try:
            %matplotlib inline
        except:
            pass
        try:
            import pandas as pd
            pd.options.display.latex.repr = True
        except:
            pass
        """
        kp.kc.execute(code, store_history=False)
        # fmt_code = '\n'.join("set_matplotlib_formats('{}')".format(fmt)
        #                      for fmt in valid_formats)
        # code = dedent(code) + fmt_code
        # kp.kc.execute(code, store_history=False)
    else:
        # raise ValueError(name)
        pass
