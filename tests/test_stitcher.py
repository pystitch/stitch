import os
import json
from textwrap import dedent

import pytest
import pypandoc

import stitch.stitch as R
from stitch.cli import enhance_args, CSS


HERE = os.path.dirname(__file__)


@pytest.fixture(scope='module')
def global_python_kernel():
    """
    A python kernel anyone can use.
    """
    return R.kernel_factory('python')


@pytest.fixture(scope='function')
def clean_python_kernel(global_python_kernel):
    """
    Takes ``global_python_kernel`` and resets all variables,
    returning the clean kernel.
    """
    R.run_code('%reset -f', global_python_kernel)
    return global_python_kernel


@pytest.fixture
def document_path():
    "Path to a markdown document"
    return os.path.join(HERE, 'data', 'small.md')


@pytest.fixture
def document():
    "In-memory markdown document"
    with open(os.path.join(HERE, 'data', 'small.md')) as f:
        doc = f.read()
    return doc


@pytest.fixture
def as_json(document):
    "JSON representation of the markdown document"
    return json.loads(pypandoc.convert_text(document, 'json', format='markdown'))


@pytest.fixture(params=['python', 'R'], ids=['python', 'R'])
def code_block(request):
    if request.param == 'python':
        code = 'def f(x):\n    return x * 2\n\nf(2)'
    elif request.param == 'R':
        code = 'f <- function(x){\n  return(x * 2)\n}\n\nf(2)'
    block = {'t': 'CodeBlock',
             'c': [['', ['{}'.format(request.param)], []],
                   code]}
    return block


@pytest.fixture
def python_kp():
    return R.kernel_factory('python')


class TestPreProcessor:

    @pytest.mark.parametrize('options, expected', [
        ('```{python}', '```{.python}'),
        ('```{r, name}', '```{.r .name}'),
        ('```{r, echo=True}', '```{.r echo=True}'),
        ('```{r, name, echo=True, eval=False}', '```{.r .name echo=True eval=False}'),
    ])
    def test_preprocess(self, options, expected):
        R.validate_options(options)
        result = R.preprocess_options(options)
        assert result == expected

    @pytest.mark.parametrize('options', [
        # '```{r name foo=bar}''',  # bad comma
        # '```{r foo=bar}'''        # no comma
        '```{r, foo=bar'''        # no curly
    ])
    def test_preprocess_raises(self, options):
        with pytest.raises(TypeError):
            R.validate_options(options)

    @pytest.mark.parametrize('kind,text,expected', [
        ("ARG", "r", ".r"),
        ("DELIM", ",", None),
        ("BLANK", " ", None),
        ("OPEN", "```{", "```{"),
        ("CLOSE", "}", "}"),
        ("KWARG", "foo=bar", "foo=bar"),
    ])
    def test_transfrom_args(self, kind, text, expected):
        result = R._transform(kind, text)
        assert result == expected

    def test_transform_raises(self):
        with pytest.raises(TypeError):
            R._transform('fake', 'foo')


class TestTesters:

    @pytest.mark.parametrize('block, expected', [
        ({'t': 'CodeBlock',
          'c': [['', ['{python}'], []],
                'def f(x):\n    return x * 2\n\nf(2)']}, True),
        ({'c': [{'c': 'With', 't': 'Str'},
                {'c': [], 't': 'Space'},
                {'c': 'options', 't': 'Str'}], 't': 'Para'}, False),
    ])
    def test_is_code_block(self, block, expected):
        result = R.is_code_block(block)
        assert result == expected

    @pytest.mark.parametrize('output, attrs, expected', [
        ([], {}, False),
        ([None], {}, False),
        ([{'text/plain': '4'}], {}, True),
        ([{'text/plain': '4'}], {'results': 'hide'}, False),
    ])
    def test_is_stitchable(self, output, attrs, expected):
        result = R.is_stitchable(output, attrs)
        assert result == expected

    @pytest.mark.parametrize('block, lang, attrs, expected', [
        ({'t': 'CodeBlock',
          'c': [['', ['{python}'], []],
                'def f(x):\n    return x * 2\n\nf(2)']}, 'python', {}, True),

        ({'c': [{'c': 'With', 't': 'Str'},
         {'c': [], 't': 'Space'},
         {'c': 'options', 't': 'Str'}], 't': 'Para'}, '', {}, False),

        ({'t': 'CodeBlock',
          'c': [['', ['{r}'], []],
                '2+2']}, 'r', {'eval': False}, False),
    ])
    def test_is_executable(self, block, lang, attrs, expected):
        result = R.is_executable(block, lang, attrs)
        assert result is expected

    @pytest.mark.parametrize('message, expected', [
        ({'content': {'name': 'stdout'}}, True),
        ({'content': {'name': 'stderr'}}, False),
        ({'content': {}}, False),
    ])
    def test_is_stdout(self, message, expected):
        result = R.is_stdout(message)
        assert result == expected

    @pytest.mark.parametrize('message, expected', [
        ({'content': {'name': 'stdout'}}, False),
        ({'content': {'name': 'stderr'}}, True),
        ({'content': {}}, False),
    ])
    def test_is_stderr(selr, message, expected):
        result = R.is_stderr(message)
        assert result == expected

    @pytest.mark.parametrize('message, expected', [
        ({'msg_type': 'execute_input'}, True),
        ({'msg_type': 'idle'}, False),
    ])
    def test_is_execute_input(selr, message, expected):
        result = R.is_execute_input(message)
        assert result == expected


class TestKernelArgs:

    @pytest.mark.parametrize('block, expected', [
        ({'t': 'CodeBlock', 'c': [['', ['python'], []], 'foo']}, 'python'),
        ({'t': 'CodeBlock', 'c': [['', ['ir'], ['foo']], 'foo']}, 'ir'),
        ({'t': 'CodeBlock', 'c': [['', ['ir'], [['foo', 'bar']]], 'foo']}, 'ir'),
    ])
    def test_extract_kernel_name(self, block, expected):
        result = R.extract_kernel_name(block)
        assert result == expected

    @pytest.mark.parametrize('block, lang, attrs, expected', [
        ({'t': 'CodeBlock',
          'c': [['', ['{python}'], []],
                'def f(x):\n    return x * 2\n\nf(2)']}, 'python', {}, True),

        ({'c': [{'c': 'With', 't': 'Str'},
         {'c': [], 't': 'Space'},
         {'c': 'options', 't': 'Str'}], 't': 'Para'}, '', {}, False),

        ({'t': 'CodeBlock',
          'c': [['', ['{r}'], []],
                '2+2']}, 'r', {'eval': False}, False),
    ])
    def test_is_executable(self, block, lang, attrs, expected):
        result = R.is_executable(block, lang, attrs)
        assert result is expected

    @pytest.mark.parametrize('code_block, expected', [
        ({'c': [['', ['python'], []], '3'], 't': 'CodeBlock'},
         (('python', None), {})),
        ({'c': [['', ['python', 'name'], []], '3'], 't': 'CodeBlock'},
         (('python', 'name'), {})),
        ({'c': [['', ['r', 'n'], [['foo', 'bar']]], '3'], 't': 'CodeBlock'},
         (('r', 'n'), {'foo': 'bar'})),
        ({'c': [['', [], [['foo', 'bar']]], '4'], 't': 'CodeBlock'},
         ((None, None), {'foo': 'bar'})),
    ])
    def test_parse_kernel_arguments(self, code_block, expected):
        result = R.parse_kernel_arguments(code_block)
        assert result == expected

    def test_parse_kernel_arguments_raises(self):
        block = {'c': [['', ['r', 'foo', 'bar'], []], '3'],
                 't': 'CodeBlock'}
        with pytest.raises(TypeError):
            R.parse_kernel_arguments(block)


class TestFormatters:

    def test_format_input(self):
        code = '2 + 2'
        expected = 'In [1]: 2 + 2'
        result = R.format_input_prompt(code, 1)
        assert result == expected

    def test_format_input_none(self):
        code = 'abcde'
        result = R.format_input_prompt(code, None)
        assert result == code

    def test_format_input_multi(self):
        code = dedent('''\
        def f(x):
            return x + 2

        f(2)
        ''').strip()
        expected = dedent('''\
        In [10]: def f(x):
            ...:     return x + 2
            ...:
            ...: f(2)
        ''').strip()
        result = R.format_input_prompt(code, 10)
        assert result == expected

    def test_wrap_input__code(self):
        block = {'t': 'code', 'c': ['a', ['b'], 'c']}
        result = R.wrap_input_code(block, None)
        assert block is not result

    @pytest.mark.parametrize('messages,expected', [
        ([{'content': {'data': {},
                       'execution_count': 4},
           'header': {'msg_type': 'execute_result'}}],
         4),

        ([{'content': {'execution_count': 2},
           'header': {'msg_type': 'execute_input'}}],
         2),

        ([{'content': {'data': {'text/plain': 'foo'}}},
          {'content': {'execution_count': 2}}],
         2),
    ])
    def test_extract_execution_count(self, messages, expected):
        assert R.extract_execution_count(messages) == expected

    @pytest.mark.parametrize('output, message, expected', [
        ([{'text/plain': '2'}],
         {'content': {'execution_count': '1'}},
         {'t': 'Div', 'c': (['', ['output'], []],
                            [{'t': 'Para',
                              'c': [{'t': 'Str',
                                     'c': 'Out[1]: 2'}]}])}),
    ])
    @pytest.mark.xfail
    def test_wrap_output(self, output, message, expected):
        result = R.wrap_output(output, message)
        assert result == expected


@pytest.mark.slow
class TestIntegration:

    def test_from_file(self, document_path):
        R.convert_file(document_path, 'html')

    def test_from_source(self, document):
        R.convert(document, 'html')

    @pytest.mark.parametrize("to, value", [
        ("html", "data:image/png;base64,"),
        ("pdf", 'unnamed_chunk_0'),  # TODO: chunk name
    ])
    def test_image(self, to, value, global_python_kernel):
        code = dedent('''\
        ```{python}
        %matplotlib inline
        import matplotlib.pyplot as plt
        plt.plot(range(4), range(4));
        ```
        ''')
        result = R.Stitch('foo', to=to).stitch(code)
        assert result[1][1]['c'][0]['t'] == 'Image'

    def test_image_chunkname(self):
        code = dedent('''\
        ```{python, chunk}
        %matplotlib inline
        import matplotlib.pyplot as plt
        plt.plot(range(4), range(4));
        ```
        ''')
        result = R.Stitch('foo', to='pdf', standalone=False).stitch(code)
        assert 'chunk' in result[1][1]['c'][0]['c'][0][0]

    def test_image_attrs(self):
        code = dedent('''\
        ```{python, chunk, width=10, height=10px}
        %matplotlib inline
        import matplotlib.pyplot as plt
        plt.plot(range(4), range(4));
        ```
        ''')
        result = R.Stitch('foo', to='html', standalone=False).stitch(code)
        attrs = result[1][1]['c'][0]['c'][0][2]
        assert ('width', '10') in attrs
        assert ('height', '10px') in attrs

    @pytest.mark.parametrize('warning, length', [
        (True, 3),
        (False, 2),
    ])
    def test_warning(self, clean_python_kernel, warning, length):
        code = dedent('''\
        ```{python}
        import warnings
        warnings.warn("Hi")
        2
        ```
        ''')
        r = R.Stitch('foo', to='html', warning=warning)
        r._kernel_pairs['python'] = clean_python_kernel
        result = r.stitch(code)
        assert len(result[1]) == length

    @pytest.mark.parametrize('to', ['latex', 'beamer'])
    def test_rich_output(self, to, clean_python_kernel):
        code = dedent('''\
        ```{python}
        import pandas as pd
        pd.options.display.latex.repr = True
        pd.DataFrame({'a': [1, 2]})
        ```
        ''')
        stitch = R.Stitch('foo', to, )
        stitch._kernel_pairs['python'] = clean_python_kernel
        meta, blocks = stitch.stitch(code)
        result = blocks[1]['c'][1]
        assert '\\begin{tabular}' in result

    def test_on_error_raises(self):
        s = R.Stitch('', on_error='raise')
        code = dedent('''\
        ```{python}
        1 / 0
        ```
        ''')
        with pytest.raises(R.StitchError):
            s.stitch(code)

        s.on_error = 'continue'
        s.stitch(code)

    @pytest.mark.parametrize('to', [
        'html', 'pdf', 'latex', 'docx',
    ])
    def test_ipython_display(self, clean_python_kernel, to):
        s = R.Stitch('', to=to)
        code = dedent('''\
        from IPython import display
        import math
        display.Markdown("$\\alpha^{pi:1.3f}$".format(pi=math.pi))
        ''')
        messages = R.run_code(code, clean_python_kernel)
        wrapped = s.wrap_output('', messages, None, clean_python_kernel, {})[0]
        assert wrapped['t'] == 'Para'
        assert wrapped['c'][0]['c'][0]['t'] == 'InlineMath'


class TestCLI:

    @pytest.mark.parametrize('expected, no_standalone, extra_args', [
        (True, False, []),
        (True, False, ['--standalone']),
        (True, False, ['-s']),
        (False, True, []),
    ])
    def test_standalone(self, expected, no_standalone, extra_args):
        args = enhance_args('', no_standalone, False, extra_args)
        result = '--standalone' in args or '-s' in args
        assert result is expected

    @pytest.mark.parametrize('expected, no_self_contained, extra_args', [
        (True, False, []),
        (True, False, ['--self-contained']),
        (False, True, []),
    ])
    def test_self_contained(self, expected, no_self_contained, extra_args):
        args = enhance_args('', False, no_self_contained, extra_args)
        result = '--self-contained' in args
        assert result is expected

    @pytest.mark.parametrize('expected, to, extra_args', [
        (['--css=%s' % CSS], 'html', []),
        (['-s', '--css=%s' % CSS], 'html', ['-s']),
        (['--css=foo.css'], 'html', ['--css=foo.css']),
        (['-c', 'foo.css'], 'html', ['-c', 'foo.css']),
    ])
    def test_css(self, expected, to, extra_args):
        result = enhance_args(to, True, True, extra_args)
        assert result == expected


@pytest.mark.slow
class TestKernel:

    def test_init_python_pre(self):
        kp = R.kernel_factory('python')
        result = R.run_code(
            'import pandas; assert pandas.options.display.latex.repr is False',
            kp)
        assert len(result) == 1

    def test_init_python_latex(self, clean_python_kernel):
        R.initialize_kernel('python', clean_python_kernel)
        result = R.run_code('assert pandas.options.display.latex.repr is False',
                            clean_python_kernel)
        assert len(result) == 2


class TestStitcher:

    def test_on_error(self):
        s = R.Stitch('')
        assert s.on_error == 'continue'
        s.on_error = 'raise'
        assert s.on_error == 'raise'

        with pytest.raises(TypeError):
            s.on_error = 'foo'

