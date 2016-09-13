import os
import click

from .stitch import convert

HERE = os.path.dirname(__file__)
CSS = os.path.join(HERE, 'static', 'default.css')


def infer_format(output_file):
    if output_file is None:
        return 'html'
    else:
        return os.path.splitext(output_file)[1].lstrip('.')


def has_css(extra_args):
    return '-c' in extra_args or '--css' in [a.split('=')[0] for a in
                                             extra_args]


def has_booktabs(extra_args):
    return 'header-includes:\\usepackage{booktabs}' in [
        '='.join(a.split('=')[1:])
        for a in extra_args
    ]


def enhance_args(to, no_standalone, no_self_contained, extra_args):
    extra_args = extra_args.copy()
    if not no_standalone and not ('-s' in extra_args or
                                  '--standalone' in extra_args):
        extra_args.append('--standalone')
    if not no_self_contained and '--self-contained' not in extra_args:
        extra_args.append('--self-contained')
    if to == 'html' and not has_css(extra_args):
        extra_args.append('--css=%s' % CSS)
    if to in ('latex', 'pdf') and not has_booktabs(extra_args):
        extra_args.append('--metadata=header-includes:\\usepackage{booktabs}')
    return extra_args


@click.command(
    context_settings=dict(ignore_unknown_options=True,
                          allow_extra_args=True)
)
@click.pass_context
@click.argument('input_file', type=click.File('rb'))
@click.option('-o', '--output_file', type=str, default=None)
@click.option('-t', '--to', default=None)
@click.option('--no-standalone', is_flag=True, default=False,
              help='Produce a document fragment instead.')
@click.option('--no-self-contained', is_flag=True, default=False,
              help='Use external files for resources like images.')
def cli(ctx, input_file, output_file, to, no_standalone, no_self_contained):
    if to is None:
        to = infer_format(output_file)
    input_text = input_file.read().decode('utf-8')
    extra_args = enhance_args(to, no_standalone, no_self_contained, ctx.args)
    convert(input_text, to, output_file=output_file, extra_args=extra_args)

if __name__ == '__main__':
    cli()
