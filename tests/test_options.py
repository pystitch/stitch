import pytest
from textwrap import dedent
from stitch.stitch import Stitch

@pytest.fixture
def doc_meta():
    data = {'date': '2016-01-01', 'title': 'My Title', 'author': 'Jack',
            'self_contained': True, 'standalone': False,
            'to': 'pdf'}
    doc = dedent('''\
    ---
    to: {to}
    title: {title}
    author: {author}
    date: {date}
    self_contained: {self_contained}
    standalone: {standalone}
    ---

    # Hi
    ''')
    return doc.format(**data), data


class TestOptions:

    def test_defaults(self):
        s = Stitch('')
        assert s.warning
        assert s.error == 'continue'
        assert s.to == 'html'
        assert s.standalone

    def test_override(self):
        doc = dedent('''\
        ---
        title: My Title
        standalone: False
        warning: False
        error: raise
        abstract: |
          This is the abstract.

          It consists of two paragraphs.
        ---

        # Hail and well met
        ''')
        s = Stitch('')
        s.stitch(doc)

        assert s.standalone is False
        assert s.warning is False
        assert s.error == 'raise'
        assert getattr(s, 'abstract', None) is None

    @pytest.mark.parametrize('key', [
        'title', 'author', 'date', 'self_contained', 'standalone', 'to'
    ])
    def test_meta(self, key, doc_meta):
        doc, meta = doc_meta
        s = Stitch('')
        s.stitch(doc)
        result = getattr(s, key)
        expected = meta[key]
        assert result == expected

@pytest.mark.slow
class TestOptionsKernel:

    def test_fig_cap(self):
        code = dedent('''\
        ```{python, fig.cap="This is a caption"}
        import matplotlib.pyplot as plt
        plt.plot(range(4), range(4))
        ```''')
        s = Stitch('')
        meta, blocks = s.stitch(code)
        result = blocks[-1]['c'][0]['c'][1][0]['c']
        assert result == 'This is a caption'



