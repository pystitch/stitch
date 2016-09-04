from .stitch import (  # noqa
    convert, convert_file, kernel_factory, run_code, Stitch
)
from .cli import cli  # noqa


from ._version import get_versions
__version__ = get_versions()['version']
del get_versions
