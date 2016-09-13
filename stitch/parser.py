"""
Chunk options-line parser.

See *Python Cookbook* 3E, recipie 2.18
"""
import re
from collections import namedtuple

Token = namedtuple("Token", ['kind', 'value'])


def validate_options(options_line):
    xpr = re.compile(r'^```{\w+.*}')
    if not xpr.match(options_line):
        raise TypeError("Invalid chunk options %s" % options_line)


def _transform(kind, text):
    if kind == 'ARG':
        result = '.' + text
    elif kind in ('DELIM', 'BLANK'):
        result = None
    elif kind in ('OPEN', 'CLOSE', 'KWARG'):
        return text
    else:
        raise TypeError('Unknown kind %s' % kind)
    return result


def tokenize(options_line):
    """
    Break an options line into a list of tokens.

    Parameters
    ----------
    options_line : str

    Returns
    -------
    tokens : list of tuples

    Notes
    -----
    The valid tokens are

      * ``KWARG``: an expression line ``foo=bar``
      * ``ARG``: a term like `python`; used for kernel & chunk names
      * ``OPEN``: The literal ``\`\`\`{``
      * ``CLOSE``: The literal ``}``
      * ``BLANK``: Whitespace
    """
    KWARG = r'(?P<KWARG>([^,=]+ *)= *(".*"|\'.*\'|[^,=}]+))'
    ARG = r'(?P<ARG>\w+)'
    OPEN = r'(?P<OPEN>```{ *)'
    DELIM = r'(?P<DELIM> *, *)'
    CLOSE = r'(?P<CLOSE>})'
    BLANK = r'(?P<BLANK>\s+)'

    master_pat = re.compile('|'.join([KWARG, ARG, OPEN, DELIM,
                                      CLOSE, BLANK]))

    def generate_tokens(pat, text):
        scanner = pat.scanner(text)
        for m in iter(scanner.match, None):
            yield Token(m.lastgroup, m.group(m.lastgroup))

    tok = list(generate_tokens(master_pat, options_line))
    return tok


def preprocess_options(options_line):
    """
    Transform a code-chunk options line to allow
    ``{python, arg, kwarg=val}`` instead of pandoc-style
    ``{.python .arg kwarg=val}``.

    Parameters
    ----------
    options_line: str

    Returns
    -------
    transformed: str
    """
    tok = tokenize(options_line)

    items = (_transform(kind, text) for kind, text in tok)
    items = filter(None, items)
    items = ' '.join(items)
    result = items.replace('{ ', '{').replace(' }', '}').replace(" {", "{")
    return result
