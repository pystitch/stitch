'''
Command line interface, built on traitlets.
'''
# Copyright (c) Jan Schulz <jasc@gmx.net>
# Copyright (c) IPython Development Team.
# Distributed under the terms of the Modified BSD License.
import logging

from jupyter_core.application import JupyterApp, base_aliases
from traitlets import (
    default, Unicode, CaselessStrEnum
)
from traitlets.config import catch_config_error

from .stitch import OUTPUT_FORMATS, convert_file

nbconvert_aliases = {}
nbconvert_aliases.update(base_aliases)
nbconvert_aliases.update({
    'to': 'StitchApp.output_format',
    'timeout': 'Knitpy.timeout',
    'output-debug': 'TemporaryOutputDocument.output_debug',
})


class StitchApp(JupyterApp):
    "Application to convert markdown documents"
    version = '0.1.0'
    name = 'stitch'
    description = Unicode(
        """Convert markdown documents, running code chunks and embedding
        the output."""
    )
    aliases = base_aliases

    # alias for to
    input_file = Unicode(help="Input file").tag(config=True)
    output_file = Unicode(help="Input file", allow_none=True).tag(config=True)

    to = CaselessStrEnum(
        OUTPUT_FORMATS,
        default_value='html',
        config=True,
        help="Format to convert to",
    ).tag(config=True)

    @default('log_level')
    def _log_level_default(self):
        return logging.INFO

    @catch_config_error
    def initialize(self, argv=None):
        super().initialize(argv)
        self.initialize_input()

    def initialize_input(self):
        self.input_file = self.extra_args[0]
        try:
            self.to = self.extra_args[1]
        except IndexError:
            self.to = None
        try:
            self.output_file = self.extra_args[2]
        except IndexError:
            self.output_file = None

    def start(self):
        super().start()
        self.convert()

    def convert(self):
        convert_file(self.input_file, self.to, output_file=self.output_file)

    def post_process(self, writer):
        pass

main = launch_new_instance = StitchApp.launch_instance
