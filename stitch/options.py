from collections.abc import Mapping
from traitlets import TraitType, Enum


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
