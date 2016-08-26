import os
import click

from .stitch import convert


def infer_format(output_file):
    if output_file is None:
        return 'html'
    else:
        return os.path.splitext(output_file)[1].lstrip('.')


@click.command(
    context_settings=dict(ignore_unknown_options=True,
                          allow_extra_args=True)
)
@click.pass_context
@click.argument('input_file', type=click.File('rb'))
@click.option('-o', '--output_file', type=str, default=None)
@click.option('-t', '--to', default=None)
def cli(ctx, input_file, output_file, to):
    if to is None:
        to = infer_format(output_file)
    input_text = input_file.read().decode('utf-8')
    convert(input_text, to, output_file=output_file, extra_args=ctx.args)

if __name__ == '__main__':
    cli()
