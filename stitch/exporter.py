import os

from nbconvert.exporters.markdown import MarkdownExporter
from traitlets import default, Unicode


class StitchExporter(MarkdownExporter):
    @default('template_file')
    def _template_file_default(self):
        return 'stitch_template'

    @property
    def template_path(self):
        return super().template_path + [os.path.join(os.path.dirname(__file__), "templates")]

    kernel_name = Unicode(default_value='python')
