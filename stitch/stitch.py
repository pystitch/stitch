"""
Read a markdown file
"""
import re
from yaml import safe_load_all
from mistune import Markdown, BlockGrammar, BlockLexer


class BlockGrammarWithOpts(BlockGrammar):
    """
    Just overriding the `fences` parser to also
    handle options. Probably a better way.
    """
    # TODO: require {} when using options and
    # delimate on that...
    fences = re.compile(
        r'^ *(`{3,}|~{3,}) *\{*(\S+)? *'  # ```lang
        r'(.*=*.*)\}*\n'  # options
        r'([\s\S]+?)\s*'
        r'\1 *(?:\n+|$)'  # ```
    )


class BlockLexerWithOpts(BlockLexer):
    """
    Again just overriding the `parse_fences` method.
    """
    grammar_class = BlockGrammarWithOpts

    def parse_fences(self, m):
        self.tokens.append({
            'type': 'code',
            'lang': m.group(2).strip(r'{}'),
            'options': m.group(3).strip('{}'),
            'text': m.group(4),
        })


OUTPUT_FORMATS = ['html', 'pdf']


def has_header(doc: str) -> bool:
    return doc.startswith('---')


def split_header(doc: str) -> (str, str):
    xpr = re.compile(r'---\n'
                     r'(?P<header>.*)'
                     r'[---\n]|[...\n]'
                     r'(?P<body>.*)',
                     re.MULTILINE | re.DOTALL)
    match = xpr.match(doc)
    return match


def tokenize(doc):
    renderer = Markdown(block=BlockLexerWithOpts)
    tok = renderer.block(doc)
    return tok


def parse_header(doc):
    return next(safe_load_all(doc))


def find_code_chunk(doc):
    pass


# ^ *(`{3,}|~{3,}) *(\S+)? *(.*=.*)*\n([\s\S]+?)\s*\1 *(?:\n+|$)k


# ^ *(`{3,}|~{3,}) *(\S+)? *\n([\s\S]+?)\s*\1 *(?:\n+|$)
