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

def test_extract_kernel_names(code_block):
    result = R.extract_kernel_names([code_block])
    assert len(result) == 1

