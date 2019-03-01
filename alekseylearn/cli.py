import click
from .alekseylearn import TrainJob
from .alekseylearn import fetch as _fetch

@click.group()
def cli():
    pass


@click.command(
    name='fit', context_settings=dict(ignore_unknown_options=True, allow_extra_args=True)
)
@click.argument('training_artifact_path')
@click.option('--driver', default='sagemaker', help='Driver to be used for running the train job.')
@click.option('--envfile', default=None, help='Code environment definition file to build with.')
@click.option('--overwrite', default=False, help='If true, overwrite existing training artifacts.')
@click.pass_context
def fit(ctx, training_artifact_path, driver, envfile, overwrite):
    config = fmt_ctx(ctx)
    overwrite = overwrite == 'True'
    j = TrainJob(training_artifact_path, driver=driver, overwrite=overwrite, config=config)
    j.fit()


@click.command(name='fetch')
@click.argument('local_path')
@click.argument('tag')
@click.argument('remote_path')
@click.option('--driver', default='sagemaker', help='Driver to be used for running the train job.')
@click.option('--extract', default='sagemaker', help='Driver to be used for running the train job.')
def fetch(local_path, tag, remote_path, driver, extract):
    _fetch(local_path, tag, remote_path, driver=driver, extract=extract)


def fmt_ctx(ctx):
    def fmt_ctx_arg(arg):
        if arg[:9] != '--config.':
            raise ValueError("Configuration arguments must be prefixed with `config`")
        return arg[9:].split('=')

    return dict([fmt_ctx_arg(arg) for arg in ctx.args])


cli.add_command(fit)
cli.add_command(fetch)