import pytest

import stitch.parser as P
from stitch.parser import Token


class TestParser:

    @pytest.mark.parametrize('options, expected', [
        ('```{python}', '```{.python}'),
        ('```{r, name}', '```{.r .name}'),
        ('```{r, echo=True}', '```{.r echo=True}'),
        ('```{r, name, echo=True, eval=False}',
         '```{.r .name echo=True eval=False}'),
        ('```{r, fig.cap="Caption"}', '```{.r fig.cap="Caption"}'),
        ('```{r, fig.cap="Cap, 2", echo=True}',
         '```{.r fig.cap="Cap, 2" echo=True}'),
        ('```{r, echo=True, fig.cap="Cap, 2"}',
         '```{.r echo=True fig.cap="Cap, 2"}'),
        ('```{r, fig.cap="Caption, too"}', '```{.r fig.cap="Caption, too"}'),
    ])
    def test_preprocess(self, options, expected):
        result = P.preprocess_options(options)
        assert result == expected

    def test_tokenize(self):
        line = '```{r, fig.width=bar}'
        result = P.tokenize(line)
        expected = [
            Token('OPEN', '```{'),
            Token('ARG', 'r'),
            Token('DELIM', ', '),
            Token('KWARG', 'fig.width=bar'),
            Token('CLOSE', '}'),
        ]

        assert result == expected

    def test_tokenize_quote(self):
        line = '```{r, fig.cap="A, Caption", echo=True}'
        result = P.tokenize(line)
        expected = [
            Token('OPEN', '```{'),
            Token('ARG', 'r'),
            Token('DELIM', ', '),
            Token('KWARG', 'fig.cap="A, Caption"'),
            Token('DELIM', ', '),
            Token('KWARG', 'echo=True'),
            Token('CLOSE', '}'),
        ]

        assert result == expected

    @pytest.mark.parametrize('kind,text,expected', [
        ("ARG", "r", ".r"),
        ("DELIM", ",", None),
        ("BLANK", " ", None),
        ("OPEN", "```{", "```{"),
        ("CLOSE", "}", "}"),
        ("KWARG", "foo=bar", "foo=bar"),
    ])
    def test_transfrom_args(self, kind, text, expected):
        result = P._transform(kind, text)
        assert result == expected

    @pytest.mark.parametrize('options', [
        # '```{r name foo=bar}''',  # bad comma
        # '```{r foo=bar}'''        # no comma
        '```{r, foo=bar'''        # no curly
    ])
    def test_preprocess_raises(self, options):
        with pytest.raises(TypeError):
            P.validate_options(options)

    def test_transform_raises(self):
        with pytest.raises(TypeError):
            P._transform('fake', 'foo')
