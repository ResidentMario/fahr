import click
from .fahr import TrainJob
from .fahr import copy_resources

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
    @click.option('--overwrite', is_flag=True, default=False,
        help='If true, overwrite existing training artifacts.')
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
    j = TrainJob(
        filepath=training_artifact_path, build_driver=build_driver, train_driver=train_driver,
        overwrite=overwrite, config=config
    )
    j.fit()


@fit_wrapper
def init(ctx, training_artifact_path, train_driver, build_driver, envfile, overwrite):
    config = fmt_ctx(ctx)
    TrainJob(
        filepath=training_artifact_path, build_driver=build_driver, train_driver=train_driver,
        overwrite=overwrite, config=config
    )


@click.command(name='status')
@click.argument('job_name')
@click.option('--train-driver', default='sagemaker',
    help='Driver to be used for running the train job.')
def status(job_name, train_driver):
    j = TrainJob(job_name=job_name, train_driver=train_driver)
    print(
        f'Training job {job_name!r} run on the {train_driver!r} service has '
        f'status {j.status()}.'
    )


@click.command(name='fetch')
@click.argument('local_path')
@click.argument('job_name')
@click.argument('remote_path', required=False)
@click.option('--train-driver', default='sagemaker',
    help='Driver to be used for running the train job.')
@click.option('--no-extract', default=False, is_flag=True,
    help='Don\'t extract the data on arrival.')
def fetch(local_path, job_name, remote_path, train_driver, no_extract):
    j = TrainJob(job_name=job_name, train_driver=train_driver)
    j.fetch(local_path, extract=not no_extract)


@click.command(name='copy')
@click.argument('src')
@click.argument('dest')
@click.option('--overwrite', is_flag=True, default=False,
    help='If true, overwrite existing job files.')
@click.option('--include-training-artifact', default=None,
    help='If set, the training artifact to copy.')
def copy(src, dest, overwrite, include_training_artifact):
    include_training_artifact = None if include_training_artifact == 'None' \
        else include_training_artifact
    copy_resources(
        src, dest, 
        overwrite=overwrite, 
        training_artifact=include_training_artifact
    )


cli.add_command(fit)
cli.add_command(status)
cli.add_command(fetch)
cli.add_command(init)
cli.add_command(copy)


__all__ = ['fit', 'status', 'fetch', 'init', 'copy']
