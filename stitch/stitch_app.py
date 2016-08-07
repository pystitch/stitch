import logging

from jupyter_core.application import JupyterApp, base_aliases
from traitlets import (
    default, Unicode, CaselessStrEnum
)
from traitlets.config import catch_config_error

from .stitch import OUTPUT_FORMATS

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
    output_format = CaselessStrEnum(
        OUTPUT_FORMATS,
        default_value='html',
        config=True,
        help="Format to convert to",
    )

    @default('log_level')
    def _log_level_default(self):
        return logging.INFO

    @catch_config_error
    def initialize(self, argv=None):
        super().initialize(argv)
        pass

    def init_documents(self):
        pass

    def init_writer(self):
        pass

    def init_postprocessor(self):
        pass

    def start(self):
        super().start()

    def init_single_document_resources(self, filename):
        pass

    def convert_single_document(self, filename):
        resources = self.init_single_document_resources(self, filename)
        output, resources = self.export_single_document(
            filename, resources
        )
        writer = self.write_single_document(output, resources)
        self.post_process(writer)

    def post_process(self, writer):
        pass

main = launch_new_instance = StitchApp.launch_instance
