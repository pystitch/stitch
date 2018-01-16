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
import mimetypes
from collections import namedtuple
from queue import Empty

from traitlets import HasTraits
from jupyter_client.manager import start_new_kernel
from nbconvert.utils.base import NbConvertBase
from pandocfilters import RawBlock, Div, CodeBlock, Image, Str, Para
import pypandoc

from .exc import StitchError
from . import options as opt
from .parser import preprocess_options

DISPLAY_PRIORITY = NbConvertBase().display_data_priority
CODE = 'code'
CODEBLOCK = 'CodeBlock'
OUTPUT_FORMATS = ['html', 'latex']
HERE = os.path.dirname(__file__)

KernelPair = namedtuple("KernelPair", "km kc")
CODE_CHUNK_XPR = re.compile(r'^```{\w+.*}|^```\w+')


class _Fig(HasTraits):
    """
    Sub-traitlet for fig related options.
    Traitlets all the way down.
    """

    width = opt.Str(None)
    height = opt.Str(None)
    cap = opt.Str(None)

# --------
# User API
# --------


class Stitch(HasTraits):
    """
    Helper class for managing the execution of a document.
    Stores configuration variables.

    Attributes
    ----------
    to : str
        The output file format. Optionally inferred by the output file
        file extension.
    title : str
        The name of the output document.
    date : str
    author : str
    self_contained : bool, default True
        Whether to publish a self-contained document, where
        things like images or CSS stylesheets are inlined as ``data``
        attributes.
    standalone : bool
        Whether to publish a standalone document (True) or fragment (False).
        Standalone documents include items like ``<head>`` elements, whereas
        non-standlone documents are just the ``<body>`` element.
    warning : bool, default True
        Whether to include text printed to stderr in the output
    error : str, default 'continue'
        How to handle exceptions in the executed code-chunks.
    prompt : str, optional
        String to put before each line of the input code. Defaults to 
        IPython-style counters. If you specify ``prompt`` option for a code
        chunk then it would have a prompt even if ``use_prompt`` is ``False``.
    echo : bool, default True
        Whether to include the input code-chunk in the output document.
    eval : bool, default True
        Whether to execute the code-chunk.

    fig.width : str
    fig.height : str

    use_prompt : bool, default False
        Whether to use prompt.
    results : str, default 'default'
        * 'default': default behaviour
        * 'pandoc': same as 'default' but some Jupyter output is parsed
          as markdown: if the output is a stdout message that is
          not warning/error or if it has 'text/plain' key.
        * 'hide': evaluate chunk but hide results

    eval_default : bool, default True
        default 'eval' attribute for every cell


    Notes
    -----
    Attirbutes can be set via the command-line, document YAML metadata,
    or (where appropriate) the chunk-options line.
    """

    # Document-traits
    to = opt.Str('html')
    title = opt.Str(None)
    date = opt.Str(None)
    author = opt.Str(None)  # TODO: Multiple authors...
    self_contained = opt.Bool(True)
    standalone = opt.Bool(True)
    use_prompt = opt.Bool(False)
    eval_default = opt.Bool(True)

    # Document or Cell
    warning = opt.Bool(True)
    error = opt.Choice({"continue", "raise"}, default_value="continue")
    prompt = opt.Str(None)
    echo = opt.Bool(True)
    eval = opt.Bool(True)
    fig = _Fig()
    results = opt.Choice({"pandoc", "hide", "default"}, default_value="default")

    def __init__(self, name: str, to: str='html',
                 standalone: bool=True,
                 self_contained: bool=True,
                 warning: bool=True,
                 error: str='continue',
                 prompt: str=None,
                 use_prompt: bool=False,
                 pandoc_extra_args: list=None):
        """
        Parameters
        ----------
        name : str
            controls the directory for supporting files
        to : str, default ``'html'``
            output format
        standalone : bool, default True
            whether to make a standalone document
        self_contained: bool, default True
        warning : bool, default True
            whether to include warnings (stderr) in the ouput.
        error : ``{"continue", "raise"}``
            how to handle errors in the code being executed.
        prompt : str, default None
        use_prompt : bool, default False
            Whether to use prompt prefixes in code chunks
        pandoc_extra_args : list of str, default None
            Pandoc extra args for converting text from markdown
            to JSON AST.
        """
        super().__init__(to=to, standalone=standalone,
                         self_contained=self_contained, warning=warning,
                         error=error, prompt=prompt, use_prompt=use_prompt)
        self._kernel_pairs = {}
        self.name = name
        self.resource_dir = self.name_resource_dir(name)
        self.pandoc_extra_args = pandoc_extra_args

    def __getattr__(self, attr):
        if '.' in attr:
            thing, attr = attr.split('.', 1)
            return getattr(getattr(self, thing), attr)
        else:
            return getattr(super(), attr)

    def has_trait(self, name):
        # intercepted `.`ed names for ease of use
        if '.' in name:
            ns, name = name.split('.', 1)
            try:
                accessor = getattr(self, ns)
            except AttributeError:
                return False
            return accessor.has_trait(name)
        else:
            return super().has_trait(name)

    def set_trait(self, name, value):
        # intercepted `.`ed names for ease of use
        if '.' in name:
            ns, name = name.split('.', 1)
            accessor = getattr(self, ns)
            return accessor.set_trait(name, value)
        else:
            return super().set_trait(name, value)

    @staticmethod
    def name_resource_dir(name):
        """
        Give the directory name for supporting resources
        """
        return '{}_files'.format(name)

    @property
    def kernel_managers(self):
        """
        dict of KernelManager, KernelClient pairs, keyed by
        kernel name.
        """
        return self._kernel_pairs

    def get_kernel(self, kernel_name):
        """
        Get a kernel from ``kernel_managers`` by ``kernel_name``,
        creating it if needed.

        Parameters
        ----------
        kernel_name : str

        Returns
        -------
        kp : KernelPair
        """
        kp = self.kernel_managers.get(kernel_name)
        if not kp:
            kp = kernel_factory(kernel_name)
            initialize_kernel(kernel_name, kp)
            self.kernel_managers[kernel_name] = kp
        return kp

    def get_option(self, option, attrs=None):
        if attrs is None:
            attrs = {}
        return attrs.get(option, getattr(self, option))

    def parse_document_options(self, meta):
        """
        Modifies self to update options, depending on the document.
        """
        for attr, val in meta.items():
            if self.has_trait(attr):
                self.set_trait(attr, val)

    def stitch(self, source: str) -> dict:
        """
        Wrapper around ``stitch_ast`` method that preprocesses
        source code to allow Stitch-style code blocks and
        then convert to loaded Pandoc JSON AST.

        Parameters
        ----------
        source : str
            the actual text to be converted

        Returns
        -------
        doc : dict
        """
        source = preprocess(source)
        ast = tokenize(source)
        return self.stitch_ast(ast)

    def stitch_ast(self, ast: dict) -> dict:
        """
        Main method for converting a document.

        Parameters
        ----------
        ast : dict
            Loaded Pandoc JSON AST

        Returns
        -------
        doc : dict
            These should be compatible with pando's JSON AST
            It's a dict with keys
              - pandoc-api-version
              - meta
              - blocks
        """
        version = ast['pandoc-api-version']
        meta = ast['meta']
        blocks = ast['blocks']

        self.parse_document_options(meta)
        lm = opt.LangMapper(meta)
        new_blocks = []

        for i, block in enumerate(blocks):
            if not is_code_block(block):
                new_blocks.append(block)
                continue
            # We should only have code blocks now...
            # Execute first, to get prompt numbers
            (lang, name), attrs = parse_kernel_arguments(block)
            if attrs.get('eval') is None:
                attrs['eval'] = self.eval_default
            kernel_name = lm.map_to_kernel(lang)
            if name is None:
                name = "unnamed_chunk_{}".format(i)
            if is_executable(block, kernel_name, attrs):
                # still need to check, since kernel_factory(lang) is executaed
                # even if the key is present, only want one kernel / lang
                kernel = self.get_kernel(kernel_name)
                messages = execute_block(block, kernel)
                execution_count = extract_execution_count(messages)
            else:
                execution_count = None
                messages = []

            # ... now handle input formatting...
            if self.get_option('echo', attrs):
                prompt = self.get_option('prompt', attrs)
                new_blocks.append(wrap_input_code(block, self.use_prompt, prompt,
                                                  execution_count, lm.map_to_style(lang)))

            # ... and output formatting
            if is_stitchable(messages, attrs):
                result = self.wrap_output(
                    name, messages, attrs,
                )
                new_blocks.extend(result)
        result = {'pandoc-api-version': version,
                  'meta': meta,
                  'blocks': new_blocks}
        return result

    def wrap_output(self, chunk_name, messages, attrs):
        """
        Wrap the messages of a code-block.

        Parameters
        ----------
        chunk_name : str
        messages : list of dicts
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
        """
        pandoc = True if (self.get_option('results', attrs) == 'pandoc') else False

        # messsage_pairs can come from stdout or the io stream (maybe others?)
        output_messages = [x for x in messages if not is_execute_input(x)]
        display_messages = [x for x in output_messages if not is_stdout(x) and
                            not is_stderr(x)]

        output_blocks = []

        # Handle all stdout first...
        for message in output_messages:
            warning = self.get_option('warning', attrs)
            if is_stdout(message) or (is_stderr(message) and warning):
                text = message['content']['text']
                output_blocks += plain_output(
                    text,
                    self.pandoc_extra_args,
                    not (is_stderr(message) and warning) and pandoc
                )

        priority = list(enumerate(NbConvertBase().display_data_priority))
        priority.append((len(priority), 'application/javascript'))
        order = dict(
            (x[1], x[0]) for x in priority
        )

        for message in display_messages:
            if message['header']['msg_type'] == 'error':
                error = self.get_option('error', attrs)
                if error == 'raise':
                    exc = StitchError(message['content']['traceback'])
                    raise exc
                blocks = plain_output(
                    '\n'.join(message['content']['traceback'])
                )
            else:
                all_data = message['content']['data']
                if not all_data:  # some R output
                    continue
                key = min(all_data.keys(), key=lambda k: order[k])
                data = all_data[key]

                if self.to in ('latex', 'pdf', 'beamer'):
                    if 'text/latex' in all_data.keys():
                        key = 'text/latex'
                        data = all_data[key]

                if key == 'text/plain':
                    # ident, classes, kvs
                    blocks = plain_output(data, self.pandoc_extra_args, pandoc)
                elif key == 'text/latex':
                    blocks = [RawBlock('latex', data)]
                elif key == 'text/html':
                    blocks = [RawBlock('html', data)]
                elif key == 'application/javascript':
                    script = '<script type=text/javascript>{}</script>'.format(
                        data)
                    blocks = [RawBlock('html', script)]
                elif key.startswith('image') or key == 'application/pdf':
                    blocks = [self.wrap_image_output(chunk_name, data, key,
                                                     attrs)]
                else:
                    blocks = tokenize_block(data, self.pandoc_extra_args)

            output_blocks += blocks
        return output_blocks

    def wrap_image_output(self, chunk_name, data, key, attrs):
        """
        Extra handling for images

        Parameters
        ----------
        chunk_name, data, key : str
        attrs: dict

        Returns
        -------
        Para[Image]
        """
        # TODO: interaction of output type and standalone.
        # TODO: this can be simplified, do the file-writing in one step
        def b64_encode(data):
            return base64.encodebytes(data.encode('utf-8')).decode('ascii')

        # TODO: dict of attrs on Stitcher.
        image_keys = {'width', 'height'}
        caption = attrs.get('fig.cap', '')

        def transform_key(k):
            # fig.width -> width, fig.height -> height;
            return k.split('fig.', 1)[-1]

        attrs = [(transform_key(k), v)
                 for k, v in attrs.items()
                 if transform_key(k) in image_keys]

        if self.self_contained:
            if 'png' in key:
                data = 'data:image/png;base64,{}'.format(data)
            elif 'svg' in key:
                data = 'data:image/svg+xml;base64,{}'.format(b64_encode(data))
            if 'png' in key or 'svg' in key:
                block = Para([Image([chunk_name, [], attrs],
                                    [Str(caption)],
                                    [data, ""])])
            else:
                raise TypeError("Unknown mimetype %s" % key)
        else:
            # we are saving to filesystem
            ext = mimetypes.guess_extension(key)
            filepath = os.path.join(self.resource_dir,
                                    "{}{}".format(chunk_name, ext))
            os.makedirs(self.resource_dir, exist_ok=True)
            if ext == '.svg':
                with open(filepath, 'wt') as f:
                    f.write(data)
            else:
                with open(filepath, 'wb') as f:
                    f.write(base64.decodebytes(data.encode('utf-8')))
            # Image :: alt text (list of inlines), target
            # Image :: Attr [Inline] Target
            # Target :: (string, string)  of (URL, title)
            block = Para([Image([chunk_name, [], []],
                                [Str(caption)],
                                [filepath, "fig: {}".format(chunk_name)])])

        return block


def convert_file(input_file: str,
                 to: str,
                 extra_args=(),
                 output_file=None) -> None:
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


def convert(source: str, to: str, extra_args=(),
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
    self_contained = '--self-contained' in extra_args
    use_prompt = '--use-prompt' in extra_args
    extra_args = [item for item in extra_args if item != '--use-prompt']
    stitcher = Stitch(name=output_name, to=to, standalone=standalone,
                      self_contained=self_contained, use_prompt=use_prompt)
    result = stitcher.stitch(source)
    result = json.dumps(result)
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
def format_input_prompt(prompt, code, number):
    """
    Format the actual input code-text.
    """
    if prompt is None:
        return format_ipython_prompt(code, number)
    lines = code.split('\n')
    formatted = '\n'.join([prompt + line for line in lines])
    return formatted


def format_ipython_prompt(code, number):
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


def wrap_input_code(block, use_prompt, prompt, execution_count, code_style=None):
    new = copy.deepcopy(block)
    code = block['c'][1]
    if use_prompt or prompt is not None:
        new['c'][1] = format_input_prompt(prompt, code, execution_count)
    if isinstance(code_style, str) and code_style != '':
        try:
            new['c'][0][1][0] = code_style
        except (KeyError, IndexError):
            pass
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


def tokenize_block(source: str, pandoc_extra_args: list=None) -> list:
    """
    Convert a Jupyter output to Pandoc's JSON AST.
    """
    if pandoc_extra_args is None:
        pandoc_extra_args = []
    json_doc = pypandoc.convert_text(source, to='json', format='markdown', extra_args=pandoc_extra_args)
    return json.loads(json_doc)['blocks']


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

    Other positional arguments are ignored by Stitch.
    All other arguments must be like ``keyword=value``.
    """
    options = block['c'][0][1]
    kernel_name = chunk_name = None
    if len(options) == 0:
        pass
    elif len(options) == 1:
        kernel_name = options[0]
    elif len(options) >= 2:
        kernel_name, chunk_name = options[0:2]
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

def plain_output(text: str, pandoc_extra_args: list=None, pandoc: bool=False) -> list:
    if pandoc:
        return tokenize_block(text, pandoc_extra_args)
    else:
        return [Div(['', ['output'], []], [CodeBlock(['', [], []], text)])]


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


def run_code(code: str, kp: KernelPair, timeout=None):
    """
    Execute a code chunk, capturing the output.

    Parameters
    ----------
    code : str
    kp : KernelPair
    timeout : int

    Returns
    -------
    outputs : List

    Notes
    -----
    See https://github.com/jupyter/nbconvert/blob/master/nbconvert
      /preprocessors/execute.py
    """
    msg_id = kp.kc.execute(code)
    while True:
        try:
            msg = kp.kc.shell_channel.get_msg(timeout=timeout)
        except Empty:
            # TODO: Log error
            raise

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


def extract_execution_count(messages):
    """
    """
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
