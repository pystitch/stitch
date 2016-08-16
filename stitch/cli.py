import click

from .stitch import convert


@click.command(
    context_settings=dict(ignore_unknown_options=True,
                          allow_extra_args=True)
)
@click.pass_context
@click.argument('input_file', type=click.File('rb'))
@click.argument('output_file', type=str, default=None)
@click.option('-t', '--to', default='html')
def cli(ctx, input_file, output_file, to):
    input_text = input_file.read().decode('utf-8')
    convert(input_text, to, output_file=output_file, extra_args=ctx.args)

if __name__ == '__main__':
    cli()
