import pytest

import stitch.parser as P
from stitch.parser import Token


class TestParser:

    @pytest.mark.parametrize('options, style, expected', [
        ('```{python}', 'default', '```{.python}'),
        ('```{r, name}', 'default', '```{.r .name}'),
        ('```{r, echo=True}', 'default', '```{.r echo=True}'),
        ('```{r, name, echo=True, eval=False}', 'default',
         '```{.r .name echo=True eval=False}'),
        ('```{r, fig.cap="Caption"}', 'default', '```{.r fig.cap="Caption"}'),
        ('```{r, fig.cap="Cap, 2", echo=True}', 'default',
         '```{.r fig.cap="Cap, 2" echo=True}'),
        ('```{r, echo=True, fig.cap="Cap, 2"}', 'default',
         '```{.r echo=True fig.cap="Cap, 2"}'),
        ('```{r, fig.cap="Caption, too"}', 'default',
         '```{.r fig.cap="Caption, too"}'),
        # simple
        ('```python', 'simple', '```{.python}'),
        ('```r, name', 'simple', '```{.r .name}'),
        ('```r, echo=True', 'simple', '```{.r echo=True}'),
        ('```r, name, echo=True, eval=False', 'simple',
         '```{.r .name echo=True eval=False}'),
        ('```r, fig.cap="Caption"', 'simple', '```{.r fig.cap="Caption"}'),
        ('```r, fig.cap="Cap, 2", echo=True', 'simple',
         '```{.r fig.cap="Cap, 2" echo=True}'),
        ('```r, echo=True, fig.cap="Cap, 2"', 'simple',
         '```{.r echo=True fig.cap="Cap, 2"}'),
        ('```r, fig.cap="Caption, too"', 'simple',
         '```{.r fig.cap="Caption, too"}'),
    ])
    def test_preprocess(self, options, style, expected):
        result = P.preprocess_options(options, style)
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

    @pytest.mark.parametrize('line, style, expected', [
        ('this is a line', P._DEFAULT, False),
        ('```python', P._DEFAULT, False),
        ('```{python}', P._DEFAULT, True),
        ('```{python, name}', P._DEFAULT, True),
        ('```{python, name, key=val}', P._DEFAULT, True),
        ('this is a line', P._SIMPLE, False),
        ('```python', P._SIMPLE, True),
        ('```{python}', P._SIMPLE, False),
        ('```{python, name}', P._SIMPLE, False),
        ('```{python, name, key=val}', P._SIMPLE, False),
    ])
    def test_is_chunk(self, line, style, expected):
        result = P.is_chunk_options(line, style)
        assert bool(result) is expected

    def test_tokenize_raises(self):
        with pytest.raises(TypeError):
            P.tokenize('{python}', 'Fake')
