from collections.abc import Mapping
from traitlets import TraitType, Enum

KERNELS = 'kernels-map'
STYLES = 'styles-map'


class Bool(TraitType):

    default_value = False
    info_text = "True or False; unwraps pandoc's JSON AST"

    def validate(self, obj, value):
        if isinstance(value, Mapping):
            assert value['t'] == 'MetaBool'
            return value['c']
        elif type(value) is bool:
            return value
        else:
            raise self.error(obj, value)


class Choice(Enum):

    info_text = "Choice from a set; unwraps pandoc's JSON AST"

    def validate(self, obj, value):
        if isinstance(value, Mapping):
            value = ' '.join(x['c'] for x in value['c'])

        if value in self.values:
            return value
        self.error(obj, value)


class Str(TraitType):

    default_value = ''
    info_text = "Choice from a set; unwraps pandoc's JSON AST"

    def validate(self, obj, value):
        if isinstance(value, Mapping):
            value = value.get('c')
            if value:
                strs = filter(None, (x.get('c') for x in value))
                value = ' '.join(strs)
        if isinstance(value, str) or value is None:
            return value
        self.error(obj, value)


class LangMapper:
    """
    Reads metadata to a dict:
    ---
    kernels-map:
      r: ir
      py: python
      py2: python2
    styles-map:
      py2: py
    ...
    then maps user specified lang names (like 'r' or 'py2')
    to kernel names that Stitch understand and
    to css classes needed for highlighing
    """
    def __init__(self, meta):
        self._kernels = self._read_dict(meta, KERNELS)
        self._styles = self._read_dict(meta, STYLES)

    @staticmethod
    def _read_dict(meta, dict_name):
        if dict_name in meta:
            try:
                ret = {key: val['c'][0]['c'] for key, val in meta[dict_name]['c'].items()}
                if all([isinstance(key, str) and isinstance(val, str) for key, val in ret.items()]):
                    return ret
            except (KeyError, IndexError, AttributeError):
                pass
            raise TypeError('Invalid {0} metadata section.'.format(dict_name))
        else:
            return {}

    def map_to_kernel(self, lang: str):
        return self._kernels.get(lang, lang)

    def map_to_style(self, lang: str):
        return self._styles.get(lang, lang)
