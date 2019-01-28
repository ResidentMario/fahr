from io import StringIO
import base64
import pathlib
import tarfile
import re
import logging
import sys

import docker
import jinja2


logger = logging.getLogger(__name__)
handler = logging.StreamHandler(sys.stdout)
logger.setLevel(logging.INFO)
handler.setLevel(logging.INFO)
handler.setFormatter(logging.Formatter('%(name)s - %(levelname)s - %(message)s'))
logger.addHandler(handler)


class TrainJob:
    def __init__(self, filepath, driver='sagemaker', envfile=None, overwrite=False, config=None):
        """
        Trains a machine learning model on the cloud.

        Parameters
        ----------
        filepath: str
            Path to the model training artifact to be executed remotely. 
            Currently only Jupyter notebooks are supported.
        driver: str
            The driver (service) that will perform the training. Currently the options are:

            * 'sagemaker' -- Launches a model training job on Amazon using AWS SageMaker.

            Future options include 'kaggle' and 'ml-engine'.
        envfile: str
            Optional path to the file that defines the environment that will be built in the image.
            Must be either a "requirements.txt" file (which will be parsed with `pip`) or an
            "environment.yml" file (which will be parsed with `conda`). If left unspecified
            whichever of these files is present in the build directory will be used.
        overwrite: bool
            If set to False `TrainJob` will respect any Dockerfile and run.sh files already present
            in the build directory. If set to True it will overwrite them.
        config: dict
            A dict of driver-specific configuration variables used to inform how the job is run.
            The precise list of configuration options differs from driver to driver. See the
            documentation for more information.
        """
        filepath = pathlib.Path(filepath).absolute()
        dirpath = filepath.parent
        if not filepath.exists():
            raise ValueError('The training artifact points to a non-existent file.')
        if filepath.suffix != '.ipynb':
            raise NotImplementedError('Currently only Jupyter notebooks are supported.')
        if driver != 'sagemaker':
            raise NotImplementedError('Currently only AWS SageMaker is supported.')

        # Check driver-specific configuration requirements
        if driver == 'sagemaker':
            if config is None or 'output_path' not in config:
                raise ValueError('The SageMaker driver requires an output_path to "s3://".')
            # The default SageMaker role enforces this rule, so we should do so too.
            if 'sagemaker' not in config['output_path']:
                raise ValueError('"output_path" must contain the word "sagemaker".')

        if envfile:
            envfile = pathlib.Path(envfile)
            if envfile.name != 'requirements.txt' and envfile.name != 'environment.yml':
                raise ValueError(
                    '"envfile" must point to "requirements.txt" or "environment.yml" file'
                )
        else:
            pip_reqfile = dirpath / 'requirements.txt'
            conda_reqfile = dirpath / 'environment.yml'

            pip_reqfile_exists = pip_reqfile.exists()
            conda_reqfile_exists = conda_reqfile.exists()

            if pip_reqfile_exists and conda_reqfile_exists:
                raise ValueError(
                    'Both requirements.txt and environment.yml are present. '
                    'Please specify which to install from using the "envfile" parameter.'
                )
            if not pip_reqfile_exists and not conda_reqfile_exists:
                raise ValueError(
                    'No requirements.txt or environment.yml present and no "envfile" specified.'
                )
            else:
                envfile = conda_reqfile if conda_reqfile_exists else pip_reqfile

        logger.info(f'Using "{envfile}" as envfile.')

        dockerfile = dirpath / 'Dockerfile'
        if not dockerfile.exists() or overwrite:
            create_dockerfile('sagemaker', dirpath, filepath.name, envfile)

        runfile = dirpath / 'run.sh'
        if not runfile.exists() or overwrite:
            create_runfile('sagemaker', dirpath, filepath)

        self.dirpath = dirpath
        self.filepath = filepath
        self.dockerfile = dockerfile
        self.runfile = runfile
        self.driver = driver
        self.config = config

        self.docker_client = docker.client.from_env()
        self.tag = f'{self.dirpath.stem}/{self.filepath.stem}'

    def build(self):
        """
        Builds the model training image locally.
        """
        path = self.dirpath.as_posix()
        logger.info(f'Building "{self.tag}" container image from "{path}".')
        self.docker_client.images.build(path=path, tag=self.tag, rm=True)

    def push(self):
        """
        Pushes the model training image to a remote repository. The repository used depends on the
        `driver`:

        * 'sagemaker' -- Pushes the image to Amazon ECR.
        """
        if self.driver == 'sagemaker':
            import boto3
            from docker.errors import ImageNotFound

            try:
                image = self.docker_client.images.get(self.tag)
            except ImageNotFound:
                raise ValueError(f'The {self.tag} image does not exist locally.')

            session = boto3.session.Session()
            sts_client = boto3.client('sts')
            ecr_client = boto3.client('ecr')
            _, account_id, repository = get_repository_info(session, self.tag, sts_client)

            # TODO: try to catch the botocore.errorfactory.RepositoryNotFound error
            try:
                ecr_client.list_images(registryId=account_id, repositoryName=self.tag)
            except:
                ecr_client.create_repository(repositoryName=self.tag)
                logger.info(
                    f'"{self.tag}" repository not found in ECR registry {account_id}. '
                    'Creating it now.'
                )
            else:
                logger.info(f'"{self.tag}" repository found in ECR registry {account_id}.')

            logger.info(f'Retrieving auth token for ECR registry {account_id}.')
            token = ecr_client.get_authorization_token(registryIds=[account_id])
            encoded_auth = token['authorizationData'][0]['authorizationToken']
            username, password = base64.b64decode(encoded_auth).decode().split(':')
            registry = token['authorizationData'][0]['proxyEndpoint']

            self.docker_client.login(username, password, registry=registry)
            image.tag(repository, 'latest')
            logger.info(f'Pushing image to ECR registry {account_id}.')
            self.docker_client.images.push(repository)

            self.session = session
            self.repository = repository
            # TODO: does reusing the client actually save a network request?
            self.sts_client = sts_client

        else:
            raise NotImplementedError

    def train(self):
        """
        Launches a remote training job. Where the job is launched depends on the `driver`:

        * sagemaker -- The job is run on an EC2 machine via the AWS SageMaker API.
        """
        if self.driver == 'sagemaker':
            import boto3
            import sagemaker as sage

            iam_client = boto3.client('iam')

            default_role_name = 'alekseylearn_sagemaker_role'
            role_name = self.config.pop('role_name', default_role_name)
            # TODO: try to catch the botocore.errorfactory.RepositoryNotFound error
            try:
                role_info = iam_client.get_role(RoleName=role_name)
            except:
                if role_name != default_role_name:
                    raise ValueError(
                        f'The {role_name} role does not exist, you must create it first.'
                    )
                else:
                    # TODO: step through the flow for creating the default if it doesn't exist
                    # https://docs.aws.amazon.com/IAM/latest/UserGuide/id_roles_create_for-user.html
                    raise NotImplementedError
                    # role = iam_client.create_role(RoleName='alekseylearn-test')

            sts_client = self.sts_client if hasattr(self, 'sts_client') else boto3.client('sts')
            logger.info(f'Assuming IAM role {role_name}.')
            assumed_role_auth = sts_client.assume_role(
                RoleArn=role_info['Role']['Arn'],
                RoleSessionName='alekseylearn_test_session'
            )
            assumed_role_session = boto3.session.Session(
                aws_access_key_id = assumed_role_auth['Credentials']['AccessKeyId'],
                aws_secret_access_key = assumed_role_auth['Credentials']['SecretAccessKey'], 
                aws_session_token = assumed_role_auth['Credentials']['SessionToken']
            )
            session = sage.Session(boto_session=assumed_role_session)
            execution_role = sage.get_execution_role(sagemaker_session=session)

            if not hasattr(self, 'repository'):
                _, _, self.repository = get_repository_info(session, self.tag, sts_client)

            train_instance_count = self.config.pop('train_instance_count', 1)
            train_instance_type = self.config.pop('train_instance_type', 'ml.c4.2xlarge')
            output_path = self.config['output_path']
            clf = sage.estimator.Estimator(
                self.repository, execution_role, 
                train_instance_count, train_instance_type,
                output_path=output_path,
                sagemaker_session=session
            )

            self.job_name = get_job_name(self.tag)
            logger.info(
                f'Fitting {self.tag} classifier with job name {self.job_name}. '
                f'Using {train_instance_count}x {train_instance_type} compute instances. '
                f'Model artifacts will be outputted to "{output_path}" S3 path.'
            )
            clf.fit(job_name=self.job_name, wait=False)
            landing_page = f'https://console.aws.amazon.com/sagemaker/'\
                'home?region=us-east-1#/jobs/{self.job_name}'
            logger.info(
                f'The training job is now running. '
                f'To track training progress visit {landing_page}. '
                f'To download finished model artifacts use `fetch` or `TrainJob.fetch`.'
            )
        else:
            raise NotImplementedError

    def fetch(self, local_path, extract=True):
        """
        Extracts the model artifacts generated by the training job to `path`.

        This method is a convenience wrapper over the `fetch` static method. Use this method to
        fetch model artifacts generated by executing `train` on the current `TrainingJob` instance.
        Use the static method to fetch model artifacts from other runs.

        Parameters
        ----------
        local_path: str or pathlib.Path
            Directory to write the model artifact to.
        extract: bool, default True
            Whether or not to untar the data on arrival.

        Raises
        ------
        ValueError -- Raised if you attempt to `fetch` without first running `fit`.
        """
        if not hasattr(self, 'job_name'):
            raise ValueError('Cannot fetch TrainJob model artifacts without fitting first.')

        return fetch(
            local_path, self.tag, self.config['output_path'], extract=extract, 
            job_name=self.job_name
        )

    def fit(self):
        """
        Executes a model training job, generating a model training artifact in a local repository.

        Parameters
        ----------
        path: str or pathlib.Path
            Directory to write the model artifact to.
        """
        self.build()
        self.push()
        self.train()


    # TODO: remote this method entirely in favor of fit-fetch?
    def run(self, path):
        """
        Executes a model training job from start to finish, generating a model training artifact 
        in a local repository.

        Parameters
        ----------
        path: str or pathlib.Path
            Directory to write the model artifact to.
        """
        path = validate_path(path)
        self.build()
        self.push()
        self.train()
        self.fetch(path)


def fetch(local_path, tag, remote_path, driver='sagemaker', extract=False, job_name=None):
    """
    Extracts model artifacts generated by a prior `TrainingJob` to `local_path`.

    Used by `TrainingJob.fetch` and by the CLI.

    Parameters
    ----------
    local_path: str or pathlib.Path
        Directory to write the model artifact to.
    tag: str
        The tag of the 
    remote_path: str
        The root directory that the model artifact got written to.
        This should be the same as the `config.output_path` of the `TrainingJob` that generated
        this model artifact.
    driver: str
        The driver (service) that will perform the training. Cf. the `TrainJob` docstring.
    extract: bool, default True
        Whether or not to untar the data on arrival.
    job_name: str or None
        The name of the job used to generate the model artifact. If omitted, the most recent model
        artifact associated with the given `tag` will be downloaded. If included, the model artifact
        associated with this specific job will be downloaded.

        This parameter is primarily intended for use by the `TrainJob.fetch` object method.
    """
    # TODO: remote_path or output_path? Standardize name.
    if driver == 'sagemaker':
        import boto3

        path = validate_path(path)

        job_name = job_name if job_name is not None else get_job_name(tag)
        output_dir = remote_path
        bucket_name = output_dir.replace('s3://', '').split('/')[0]
        bucket_path = '/'.join(output_dir.replace('s3://', '').split('/')[1:])
        model_path = f'{bucket_path}{job_name}/output/model.tar.gz'
        local_model_filepath = pathlib.Path(f'{path}/model.tar.gz').absolute().as_posix()

        s3_client = boto3.client('s3')
        logger.info(f'Downloading model artifact to "{local_model_filepath}".')
        s3_client.download_file(bucket_name, model_path, local_model_filepath)

        if extract:
            tarfile.open(local_model_filepath).extractall()
            pathlib.Path(local_model_filepath).unlink()

    else:
        raise NotImplementedError


def create_template(template_name, dirpath, filename, **kwargs):
    """
    Helper function for writing a parameterized template to disk.
    """
    template_env = jinja2.Environment(loader=jinja2.PackageLoader('alekseylearn', 'templates'))
    template_text = template_env.get_template(template_name).render(
        **kwargs
    )
    with open(dirpath / filename, 'w') as f:
        f.write(template_text)


def create_dockerfile(driver, dirpath, filepath, envfile):
    """
    Creates a Dockerfile compatible with the given `driver` and writes to to disk.

    Parameters
    ----------
    driver: str
        The driver (service) that will perform the training.
    dirpath: str
        Path to the directory being bundled.
    filepath: str
        Name of the model training artifact being bundled.
    """
    if driver == 'sagemaker':
        create_template(
            'sagemaker/Dockerfile', dirpath, 'Dockerfile', filepath=filepath,
            envfile=str(envfile)
        )
    else:
        raise NotImplementedError


def create_runfile(driver, dirpath, filepath):
    """
    Creates a run.sh entrypoint for the Docker image compatible with the given `driver`, and 
    writes it to disk.
    """
    if driver == 'sagemaker':
        if filepath.suffix == '.ipynb':
            run_cmd =\
                "jupyter nbconvert --execute --ExecutePreprocessor.timeout=-1 "\
                "--to notebook --inplace build.ipynb"
            create_template('sagemaker/run.sh', dirpath, 'run.sh', run_cmd = run_cmd)
        else:
            raise NotImplementedError
    else:
        raise NotImplementedError


def get_repository_info(session, tag, sts_client=None):
    sts_client = sts_client if sts_client else session.client('sts')
    account_id = sts_client.get_caller_identity()['Account']
    region = session.region_name

    repository = f'{account_id}.dkr.ecr.{region}.amazonaws.com/{tag}'
    return region, account_id, repository


def get_job_name(tag):
    import boto3
    
    # FIXME: support paginated requests, as otherwise most recent run can fall of the list
    sagemaker_client = boto3.client('sagemaker')
    finished_jobs = sagemaker_client.list_training_jobs()['TrainingJobSummaries']
    # TODO: ensure that the job name is always a valid ARN name
    # re.compile('^[a-zA-Z0-9](-*[a-zA-Z0-9])*').matchall(prefix) is not None
    prefix = f'alekseylearn-{tag.replace("/", "-")}'
    previous_jobs = [j for j in finished_jobs if j['TrainingJobName'].startswith(prefix)]
    n = len(previous_jobs)
    job_name = f'{prefix}-{n}'
    return job_name


def validate_path(path):
    path = pathlib.Path(path)
    if not path.exists():
        raise ValueError("Output parameter must point to an existing directory.")
    if not path.is_dir():
        raise ValueError("Output path must be a directory.")
    return path

__all__ = ['TrainJob', 'fetch']
