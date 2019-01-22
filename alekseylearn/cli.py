import click
from .alekseylearn import TrainJob

@click.group()
def cli():
    pass


@click.command(
    name='run', context_settings=dict(ignore_unknown_options=True, allow_extra_args=True)
)
@click.argument('training_artifact_path')
@click.argument('model_artifact_path')
@click.option('--driver', default='sagemaker', help='Driver to be used for running the train job.')
@click.option('--overwrite', default=False, help='If true, overwrite existing training artifacts.')
@click.pass_context
def run(ctx, training_artifact_path, model_artifact_path, driver, overwrite):
    def fmt_ctx_arg(arg):
        if arg[:9] != '--config.':
            raise ValueError("Configuration arguments must be prefixed with `config`")
        return arg[9:].split('=')

    config = dict([fmt_ctx_arg(arg) for arg in ctx.args])
    j = TrainJob(training_artifact_path, driver=driver, overwrite=overwrite, config=config)
    j.run(model_artifact_path)


cli.add_command(run)