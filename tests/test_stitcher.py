import os
import json
from textwrap import dedent

import pytest
import pypandoc

import stitch.stitch as R


HERE = os.path.dirname(__file__)


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
        '```python',              # no curly
        # '```{r foo=bar}'''        # no comma
        '```{r, foo=bar'''        # no curly
    ])
    def test_preprocess_raises(self, options):
        with pytest.raises(TypeError):
            R.validate_options(options)

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

    def test_extract_kernel_name(self, code_block):
        result = R.extract_kernel_name(code_block)
        assert result in ('R', 'python')

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


class TestKernelArgs:

    @pytest.mark.parametrize('block, expected', [
        ({'t': 'CodeBlock', 'c': [['', ['python'], []], 'foo']}, 'python'),
        ({'t': 'CodeBlock', 'c': [['', ['ir'], ['foo']], 'foo']}, 'ir'),
        ({'t': 'CodeBlock', 'c': [['', ['ir'], [['foo', 'bar']]], 'foo']}, 'ir'),
    ])
    def test_extract_kernel_name(self, block, expected):
        result = R.extract_kernel_name(block)
        assert result == expected


class TestFormatters:

    def test_format_input(self):
        code = '2 + 2'
        expected = 'In [1]: 2 + 2'
        result = R.format_input_prompt(code, 1)
        assert result == expected

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

    def test_image(self, python_kp):
        code = dedent('''\
        # Testing

        ```{python}
        %matplotlib inline
        import matplotlib.pyplot as plt
        plt.plot(range(4), range(4))
        ```
        ''')
        result = R.stitch(code)
        assert "data:image/png;base64," in result

