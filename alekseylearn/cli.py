import click
from .alekseylearn import TrainJob
from .alekseylearn import fetch as _fetch

@click.group()
def cli():
    pass


def fit_wrapper(func):
    @click.command(
        name=func.__name__, context_settings=dict(ignore_unknown_options=True, allow_extra_args=True)
    )
    @click.argument('training_artifact_path')
    @click.option('--train-driver', default='sagemaker',
        help='Driver to be used for running the train job.')
    @click.option('--build-driver', default='local',
        help='Driver to be used for building the model image.')
    @click.option('--envfile', default=None, help='Code environment definition file to build with.')
    @click.option('--overwrite', default=False, help='If true, overwrite existing training artifacts.')
    @click.pass_context
    def new_func(*args, **kwargs):
        func(*args, **kwargs)
    
    return new_func


def fmt_ctx(ctx):
    def fmt_ctx_arg(arg):
        if arg[:9] != '--config.':
            raise ValueError(
                f'Invalid input argument "{arg}"; '
                f'if this is a configuration argument it must be prefixed with "config".'
            )
        return arg[9:].split('=')

    return dict([fmt_ctx_arg(arg) for arg in ctx.args])


@fit_wrapper
def fit(ctx, training_artifact_path, train_driver, build_driver, envfile, overwrite):
    config = fmt_ctx(ctx)
    overwrite = overwrite == 'True'
    j = TrainJob(
        training_artifact_path, build_driver=build_driver, train_driver=train_driver,
        overwrite=overwrite, config=config
    )
    j.fit()


@fit_wrapper
def init(ctx, training_artifact_path, train_driver, build_driver, envfile, overwrite):
    config = fmt_ctx(ctx)
    overwrite = overwrite == 'True'
    TrainJob(
        training_artifact_path, build_driver=build_driver, train_driver=train_driver,
        overwrite=overwrite, config=config
    )


@click.command(name='fetch')
@click.argument('local_path')
@click.argument('tag')
@click.argument('remote_path')
@click.option('--driver', default='sagemaker', help='Driver to be used for running the train job.')
@click.option('--extract', default=True, help='Whether or not to untar the data on arrival.')
def fetch(local_path, tag, remote_path, driver, extract):
    extract = extract == 'True'
    _fetch(local_path, tag, remote_path, train_driver=driver, extract=extract)


cli.add_command(fit)
cli.add_command(fetch)
cli.add_command(init)