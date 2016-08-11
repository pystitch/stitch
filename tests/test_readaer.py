import os
import json
import pytest
import pypandoc

import stitch.stitch as R


HERE = os.path.dirname(__file__)


@pytest.fixture
def document():
    with open(os.path.join(HERE, 'data', 'test1.md')) as f:
        doc = f.read()
    return doc


@pytest.fixture
def as_json(document):
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

@pytest.mark.parametrize('block, expected', [
    ({'t': 'CodeBlock',
      'c': [['', ['{python}'], []],
            'def f(x):\n    return x * 2\n\nf(2)']}, True),
    ({'c': [{'c': 'With', 't': 'Str'},
     {'c': [], 't': 'Space'},
     {'c': 'options', 't': 'Str'}], 't': 'Para'}, False),
])
def test_to_execute(block, expected):
    result = R.to_execute(block)
    assert result is expected

def test_extract_kernel_name(code_block):
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
def test_parse_kernel_arguments(code_block, expected):
    result = R.parse_kernel_arguments(code_block)
    assert result == expected


@pytest.mark.parametrize('output, expected', [
    ([{'text/plain': '2'}],
     {'t': 'Div', 'c': (['', ['output'], []],
                        [{'t': 'Para', 'c': [{'t': 'Str', 'c': '2'}]}])}),
])
def test_wrap_output(output, expected):
    result = R.wrap_output(output)
    assert result == expected

