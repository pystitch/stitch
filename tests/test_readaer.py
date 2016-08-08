import pytest

import stitch.reader as R


def test_regex_matches():
    s = '```{python}\ndef f(x):\n    return x\n```'
    assert R.BlockGrammarWithOpts.fences.match(s)


@pytest.mark.parametrize('options,expected_options', [
    ('python', ''),
    ('{python}', ''),
    ('python echo=False execute=True', 'echo=False execute=True'),
])
def test_lexer(options, expected_options):
    doc = '```{}\ndef f(x):\n    return x\n```'.format(options)
    tok = R.tokenize(doc)
    expected = [{
        'type': 'code',
        'lang': 'python',
        'options': expected_options,
        'text': 'def f(x):\n    return x',
    }]
    assert tok == expected


def test_lexer_opts():
    doc = '```{python echo=False}\ndef f(x):\n    return x\n```'
    tok = R.tokenize(doc)
    assert tok[0]['type'] == 'code'


@pytest.mark.parametrize('endmarker', ['---\n', '...\n'])
def test_parse_header(endmarker):
    doc = '---\ntitle: test title\nauthor: test author\n' + endmarker
    result = R.parse_header(doc)
    assert result == {'title': 'test title', 'author': 'test author'}

