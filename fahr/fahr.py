from io import StringIO
import base64
import pathlib
import tarfile
import re
import logging
import sys
import shutil
import json
import ast
import subprocess

import docker
import jinja2


logger = logging.getLogger(__name__)
if not logger.handlers:
    handler = logging.StreamHandler(sys.stdout)
    logger.setLevel(logging.INFO)
    handler.setLevel(logging.INFO)
    handler.setFormatter(logging.Formatter('%(name)s - %(levelname)s - %(message)s'))
    logger.addHandler(handler)

class TrainJob:
    def __init__(
        self, filepath, build_driver='local', train_driver='sagemaker', 
        envfile=None, overwrite=False, config=None,
    ):
        """
        Trains a machine learning model on the cloud.

        Parameters
        ----------
        filepath: str
            Path to the model training artifact to be executed remotely. 
            Currently only Jupyter notebooks are supported.
        build_driver: str
            The driver (service) that will build the model training image. The options are:

            * 'local' -- Builds the model container on your local machine w/o GPU.
            * 'local-gpu' -- Builds the model container on your local machine w/ GPU.
        
            Note that if you are using the Kaggle training driver, no build driver needs to be 
            specified.
        train_driver: str
            The driver (service) that will perform the training. Currently the options are:

            * 'sagemaker' -- Launches a model training job on Amazon using AWS SageMaker.
            * 'kaggle' -- Launches a model training job on Kaggle using Kaggle Kernels.

            Future options include 'ml-engine'.
        envfile: str
            Optional path to the file that defines the environment that will be built in the image.
            Must be a "requirements.txt" file (which will be parsed with `pip`). If left unspecified
            the file present in the build directory will be used.
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
        if filepath.suffix != '.ipynb' and filepath.suffix != '.py':
            raise NotImplementedError(
                'Currently only Jupyter notebooks and Python scripts are supported.'
            )
        if build_driver != 'local' and build_driver != 'local-gpu':
            raise NotImplementedError('Currently only local Docker builds are supported.')

        # Check driver-specific configuration requirements
        if train_driver == 'sagemaker':
            if config is None or 'output_path' not in config:
                raise ValueError('The SageMaker driver requires an output_path to "s3://".')

            # Ensure that the job name is always a valid ARN name
            tag = f'{dirpath.stem}/{filepath.stem}'.replace("/", "-").replace("_", "-")
            regex = '^[a-zA-Z0-9](-*[a-zA-Z0-9])*'
            try:
                assert re.match(regex, tag).span()[1] == len(tag)
            except (AssertionError, AttributeError):
                raise ValueError(f'"File name must satisfy regex "{regex}".')

        elif train_driver == 'kaggle':
            if config is None or 'username' not in config:
                raise ValueError('The Kaggle driver requires a username.')

            # Ensure that the job name is always a valid kernel name
            # TODO: we require 5+ alphabetical for now, but should relax this restriction
            filename = f'{filepath.stem}'.replace("/", "-").replace("_", "-")
            regex = '^[a-zA-Z]{5,}'
            try:
                re.match(regex, filename).span()[1] == len(filename)
            except (AssertionError, AttributeError):
                raise ValueError(f'"File name must satisfy regex "{regex}".')

            tag = f'{config["username"]}/{filename}'

            # Ensure that the input sources are valid.
            for source_param in ['dataset_sources', 'kernel_sources', 'competition_sources']:
                if source_param in config:
                    source_val = config[source_param]
                    try:
                        parsed_source_val = ast.literal_eval(source_val)
                        assert isinstance(parsed_source_val, list)
                        config[source_param] = parsed_source_val
                    except (ValueError, AssertionError):
                        raise ValueError(
                            f'Invalid input for "{source_param}": "{config[source_param]}" '
                            f'is not a list.'
                        )

            # Ensure that the input source actually exist.
            import pdb; pdb.set_trace()
            for dataset in config.get('dataset_sources', []):
                if subprocess.run(
                    ["kaggle", "datasets", "status", dataset],
                    stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT
                ).returncode != 0:
                    raise ValueError(
                        f'Invalid input: the dataset "{dataset}" does not exist.'
                    )
            for kernel in config.get('kernel_sources', []):
                if subprocess.run(
                    ["kaggle", "kernels", "status", kernel],
                    stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT
                ).returncode != 0:
                    raise ValueError(
                        f'Invalid input: the kernel "{dataset}" does not exist.'
                    )

            if 'competition_sources' in config:
                current_competitions = str(subprocess.check_output(
                    ["kaggle", "competitions", "list"],
                    stderr=subprocess.STDOUT
                ))
                for competition in config.get('competition_sources'):
                    if competition not in current_competitions:
                        raise ValueError(
                            f'Invalid input: the competition "{competition}" does not exist '
                            f'or is not currently running.'
                        )

        else:
            raise NotImplementedError(
                'Currently only AWS SageMaker and Kaggle Kernels are supported.'
            )

        if train_driver != 'kaggle':
            if envfile:
                envfile = pathlib.Path(envfile)
                if envfile.name != 'requirements.txt':
                    raise ValueError(
                        '"envfile" must point to "requirements.txt" file'
                    )
            else:
                pip_reqfile = dirpath / 'requirements.txt'
                pip_reqfile_exists = pip_reqfile.exists()
                if not pip_reqfile_exists:
                    raise ValueError('No requirements.txt present and no "envfile" specified.')
                else:
                    envfile = pip_reqfile

            logger.info(f'Using "{envfile}" as envfile.')

            envfile = envfile.absolute().relative_to(pathlib.Path.cwd()).as_posix()
            dockerfile, runfile = create_resources(
                build_driver, train_driver, dirpath, filepath, envfile, overwrite
            )
        else:  # train_driver == 'kaggle':
            # Kaggle kernels run in a default environment. The web UI allows you to 
            # specify custom packages but the API does not currently support this feature.
            if envfile:
                raise ValueError(
                    'The "envfile" parameter is specified but shouldn\'t be.'
                    'Kaggle does not currently support running code in custom containers'
                    'in the API.'
                )
            create_kaggle_resources(
                dirpath, filepath, tag,
                title=config.pop('title', filename.replace('_', ' ').replace('-', ' ').title()),
                is_private=config.pop('is_private', False),
                enable_gpu=config.pop('enable_gpu', False),
                enable_internet=config.pop('enable_internet', False),
                dataset_sources=config.pop('dataset_sources', []),
                kernel_sources=config.pop('kernel_sources', []),
                competition_sources=config.pop('competition_sources', []),
            )

        self.dirpath = dirpath
        self.filepath = filepath
        self.build_driver = build_driver
        self.train_driver = train_driver
        self.config = config
        self.tag = tag

        if train_driver != 'kaggle':
            self.dockerfile = dockerfile
            self.runfile = runfile
            self.docker_client = docker.client.from_env()

    def build(self):
        """
        Builds the model training image locally.
        """
        if self.train_driver != 'kaggle':
            if self.build_driver == 'local' or self.build_driver == 'local-gpu':
                path = self.dirpath.as_posix()
                logger.info(f'Building "{self.tag}" container image from "{path}".')
                self.docker_client.images.build(path=path, tag=self.tag, rm=True)
            else:
                raise NotImplementedError

    def push(self):
        """
        Pushes the model training image to a remote repository. The repository used depends on the
        `train_driver`:

        * 'sagemaker' -- Pushes the image to Amazon ECR.
        """
        if self.train_driver == 'sagemaker':
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

        elif self.train_driver == 'kaggle':
            return

        else:
            raise NotImplementedError

    def train(self):
        """
        Launches a remote training job. Where the job is launched depends on the `train_driver`:

        * sagemaker -- The job is run on an EC2 machine via the AWS SageMaker API.
        * kaggle -- The job is run in a Kaggle Kernel via the Kaggle API.
        """
        if self.train_driver == 'sagemaker':
            import boto3
            import sagemaker as sage

            iam_client = boto3.client('iam')

            default_role_name = 'fahr_sagemaker_role'
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
                    # role = iam_client.create_role(RoleName='fahr-test')

            # if the current execution context is not the `role_name` role, assume it
            sts_client = self.sts_client if hasattr(self, 'sts_client') else boto3.client('sts')
            current_execution_context_arn = sts_client.get_caller_identity()['Arn']

            if f'assumed-role/{role_name}' in current_execution_context_arn:
                logger.info(
                    f'{role_name} is both the desired IAM role and the current execution context. '
                    f'Creating a new session using the current session authorization credentials.'
                )
                aws_access_key_id, aws_secret_access_key, aws_session_token =\
                    self.session.get_credentials().get_frozen_credentials()
            else:
                logger.info(f'Assuming IAM role {role_name}.')
                auth = sts_client.assume_role(
                    RoleArn=role_info['Role']['Arn'],
                    RoleSessionName='fahr_session'
                )['Credentials']
                aws_access_key_id, aws_secret_access_key, aws_session_token = (
                    auth['AccessKeyId'], auth['AccessKeyId'], auth['SessionToken']
                )

            assumed_role_session = boto3.session.Session(
                aws_access_key_id=aws_access_key_id,
                aws_secret_access_key=aws_secret_access_key, 
                aws_session_token=aws_session_token
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

            self.job_name = get_next_job_name(self.tag)
            logger.info(
                f'Fitting {self.tag} training job with job name {self.job_name}. '
                f'Using {train_instance_count}x {train_instance_type} compute instances.'
            )
            clf.fit(job_name=self.job_name, wait=False)
            landing_page = (
                f'https://console.aws.amazon.com/sagemaker/'
                f'home?#/jobs/{self.job_name}'
            )
            logs_page = (
                f'https://console.aws.amazon.com/cloudwatch/'
                f'home?#logStream:group=/aws/sagemaker/TrainingJobs;'
                f'streamFilter=typeLogStreamPrefix'
            )
            download_cmd = f'fahr fetch --driver="sagemaker" ./ "{self.tag}" "{output_path}"'
            logger.info(
                f'The training job is now running. '
                f'To track training progress visit {landing_page}. '
                f'To see training logs visit {logs_page}. '
                f'To download finished model artifacts run {download_cmd} after '
                f'training is complete.'
            )

        elif self.train_driver == 'kaggle':
            logger.info(
                # TODO: get and display the number of the run, e.g. version 3, 4, etc.
                f'Fitting {self.tag} training job.'
            )
            subprocess.run(
                ["kaggle", "kernels", "push", "-p", self.dirpath.as_posix()],
                stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT
            )
            download_cmd = f'fahr fetch --driver="kaggle" ./ "{self.tag}"'
            landing_page = f'https://www.kaggle.com/{self.tag}'
            logger.info(
                f'The training job is now running. '
                f'To track training progress visit {landing_page}. '
                f'To download finished model artifacts run {download_cmd} after '
                f'training is complete.'
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
            local_path, self.tag, self.config['output_path'], 
            extract=extract, job_name=self.job_name
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


def fetch(local_path, tag, remote_path, train_driver='sagemaker', extract=False, job_name=None):
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
    train_driver: str
        The train_driver (service) that will perform the training. Cf. the `TrainJob` docstring.
    extract: bool, default True
        Whether or not to untar the data on arrival.
    job_name: str or None
        The name of the job used to generate the model artifact. If omitted, the most recent model
        artifact associated with the given `tag` will be downloaded. If included, the model artifact
        associated with this specific job will be downloaded.

        This parameter is primarily intended for use by the `TrainJob.fetch` object method.
    """
    # TODO: remote_path or output_path? Standardize name.
    if train_driver == 'sagemaker':
        import boto3

        path = validate_path(local_path)
        job_name = job_name if job_name is not None else get_previous_job_name(tag)
        output_dir = remote_path
        bucket_name = output_dir.replace('s3://', '').split('/')[0]
        bucket_path = '/'.join(output_dir.replace('s3://', '').split('/')[1:])
        model_path = f'{bucket_path}{job_name}/output/model.tar.gz'
        local_model_filepath = pathlib.Path(f'{path}/model.tar.gz').absolute()
        local_model_filepath_str = local_model_filepath.as_posix()
        local_model_directory_str = f'{local_model_filepath.parent.as_posix()}/'

        s3_client = boto3.client('s3')
        s3_client.download_file(bucket_name, model_path, local_model_filepath_str)

        if extract:
            tarfile.open(local_model_filepath_str).extractall()
            pathlib.Path(local_model_filepath_str).unlink()
            logger.info(f'Downloaded model artifact(s) to "{local_model_directory_str}".')
        else:
            logger.info(f'Downloaded model artifact(s) to "{local_model_filepath_str}".')

    elif train_driver == 'kaggle':
        path = validate_path(local_path)
        subprocess.run(
            ["kaggle", "kernels", "output", tag, "-p", local_path],
            stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT
        )
        logger.info(f'Downloaded model artifact(s) to "{local_path}".')

    else:
        raise NotImplementedError


def copy_resources(src, dest, overwrite=True, training_artifact=None):
    """
    Copies training file resources from `src` to `dest`.

    Parameters
    ----------
    src: str
        The source directory.
    dest: str
        The target directory.
    overwrite: bool, default True
        Whether or not to overwrite existing resources in the target directory.
    training_artifact: bool, default False
        Whether or not to include the training artifact in the list of files copied.
    """
    src, dest = pathlib.Path(src), pathlib.Path(dest)

    if not (src / 'Dockerfile').exists():
        raise ValueError('No "Dockerfile" found in the source directory.')
    if not (src / 'run.sh').exists():
        raise ValueError('No "run.sh" entrypoint found in the source directory.')
    if not (src / 'requirements.txt').exists():
        raise ValueError('No "requirements.txt" found in the source directory.')        
    if training_artifact is not None and not (src / training_artifact).exists():
        raise ValueError(
            f'Training artifact "{training_artifact}" not found in source directory.'
        )

    if not (dest / 'Dockerfile').exists() or overwrite:
        shutil.copy(str(src / 'Dockerfile'), str(dest / 'Dockerfile'))
    if not (dest / 'run.sh').exists() or overwrite:
        shutil.copy(str(src / 'run.sh'), str(dest / 'run.sh'))
    if not (dest / 'requirements.txt').exists() or overwrite:
        shutil.copy(str(src / 'requirements.txt'), str(dest / 'requirements.txt'))
    if (training_artifact is not None
        and (not (dest / training_artifact).exists() or overwrite)):
        shutil.copy(str(src / training_artifact), str(dest / training_artifact))
    
    logger.info(f'Copied files over to "{dest}".')


def create_template(template_name, dirpath, filename, **kwargs):
    """
    Helper function for writing a parameterized template to disk.
    """
    template_env = jinja2.Environment(loader=jinja2.PackageLoader('fahr', 'templates'))
    template_text = template_env.get_template(template_name).render(
        **kwargs
    )
    with open(dirpath / filename, 'w') as f:
        f.write(template_text)


def create_dockerfile(build_driver, train_driver, dirpath, filepath, envfile):
    """
    Creates a Dockerfile compatible with the given drivers and writes to to disk.

    Parameters
    ----------
    build_driver: str
        The build_driver (service) that will build the model image.
    train_driver: str
        The train_driver (service) that will perform the training.
    dirpath: str
        Path to the directory being bundled.
    filepath: str
        Name of the model training artifact being bundled.
    envfile: str
        Path to the environment file that will be built in the image.
    """
    if train_driver == 'sagemaker':
        create_template(
            'sagemaker/Dockerfile.template', dirpath, 'Dockerfile', 
            filepath=filepath, envfile=envfile, build_driver=build_driver
        )
    else:
        raise NotImplementedError


def create_runfile(build_driver, train_driver, dirpath, filepath):
    """
    Creates a run.sh entrypoint for the Docker image compatible with the given `train_driver`,
    and writes it to disk.

    Parameters
    ----------
    build_driver: str
        The build_driver (service) that will build the model image.
    train_driver: str
        The train_driver (service) that will perform the training.
    dirpath: str
        Path to the directory being bundled.
    filepath: str
        Name of the model training artifact being bundled.
    """
    if train_driver == 'sagemaker':
        if filepath.suffix == '.ipynb':
            run_cmd =\
                (f"jupyter nbconvert --execute --ExecutePreprocessor.timeout=-1 "
                 f"--to notebook --inplace {filepath.name}")
            create_template('sagemaker/run.template', dirpath, 'run.sh', run_cmd=run_cmd)
        elif filepath.suffix == '.py':
            run_cmd =\
                f"python {filepath.name}"
            create_template('sagemaker/run.template', dirpath, 'run.sh', run_cmd=run_cmd)                
        else:
            raise NotImplementedError
    else:
        raise NotImplementedError


def create_resources(build_driver, train_driver, dirpath, filepath, envfile, overwrite):
    """
    Creates file resources that will be used by this library in a target directory 
    (specifically, a Dockerfile and a run.sh entrypoint). Calls `create_runfile` and
    `create_dockerfile` as a subroutine.

    See also: create_kaggle_resources.

    Parameters
    ----------
    build_driver: str
        The build_driver (service) that will build the model image.
    train_driver: str
        The train_driver (service) that will perform the training.
    dirpath: str
        Path to the directory being bundled.
    filepath: str
        Name of the model training artifact being bundled.
    envfile: str
        Path to the environment file that will be built in the image.
    overwrite: bool

    Returns
    -------
    dockerfile, runfile: tuple
        The filepaths written to.
    """
    dockerfile = dirpath / 'Dockerfile'
    dockerfile_exists = dockerfile.exists()
    if dockerfile_exists and not overwrite:
        logger.info(
            f'A Dockerfile already exists at "{dockerfile}" and "overwrite" is set to False. '
            f'The existing Dockerfile will be reused.'
        )
    elif dockerfile_exists and overwrite:
        logger.info(
            f'A Dockerfile already exists at "{dockerfile}" and "overwrite" is set to True. '
            f'Overwriting the file that is currently there.'
        )
        create_dockerfile(build_driver, train_driver, dirpath, filepath.name, envfile)
    else:  # dockerfile_exists
        logger.info(f'Creating new Dockerfile at "{dockerfile}".')
        create_dockerfile(build_driver, train_driver, dirpath, filepath.name, envfile)

    runfile = dirpath / 'run.sh'
    runfile_exists = runfile.exists()
    if runfile_exists and not overwrite:
        logger.info(
            f'An image entrypoint already exists at "{runfile}" and "overwrite" is set to False. '
            f'The existing file will be reused.'
        )
    elif runfile_exists and overwrite:
        logger.info(
            f'An image entrypoint already exists at "{runfile}" and "overwrite" is set to True. '
            f'Overwriting the file that is currently there.'
        )
        create_runfile(build_driver, train_driver, dirpath, filepath)
    else:  # runfile_exists
        logger.info(f'Creating new image entrypoint at "{runfile}".')
        create_runfile(build_driver, train_driver, dirpath, filepath)  

    return dockerfile, runfile


def create_kaggle_resources(
        dirpath, filepath, tag, title, is_private, enable_gpu, enable_internet,
        dataset_sources, kernel_sources, competition_sources
    ):
    """
    Create resources necessary for a Kaggle Kernels run, specifically, a 'kernel-metadata.json'
    file.
    
    See also: create_resources.

    Parameters
    ----------
    dirpath: str
        Path to the directory containing the training artifact.
    filepath: str
        Path to the training artifact.
    tag: str
        The 'username/kernel_slug' pair that uniquely identifies this kernel.

    Returns
    -------
    None
    """
    kernel_metadata = {
        'id': f'{tag}',
        'title': title,
        'code_file': filepath.as_posix(),
        'language': 'python',
        'kernel_type': 'notebook' if filepath.suffix == '.ipynb' else 'script',
        'is_private': is_private,
        'enable_gpu': enable_gpu,
        'enable_internet': enable_internet,
        'dataset_sources': dataset_sources,
        'competition_sources': competition_sources,
        'kernel_sources': kernel_sources
    }
    kernel_metadata_filepath = (dirpath / 'kernel-metadata.json').as_posix()
    logger.info(f'Writing kernel metadata to "{kernel_metadata_filepath}".')
    with open(kernel_metadata_filepath, 'w') as fp:
        json.dump(kernel_metadata, fp)


def get_repository_info(session, tag, sts_client=None):
    """
    Identify the ECR repository for a given tag. SageMaker only.
    """
    sts_client = sts_client if sts_client else session.client('sts')
    account_id = sts_client.get_caller_identity()['Account']
    region = session.region_name

    repository = f'{account_id}.dkr.ecr.{region}.amazonaws.com/{tag}'
    return region, account_id, repository


def get_previous_job_name(tag):
    """
    Given the training run tag, determine the name of the job immediately previous to this
    one, assuming one exists. SageMaker only.
    """
    import boto3
    
    # FIXME: support paginated requests, as otherwise most recent run can fall of the list
    sagemaker_client = boto3.client('sagemaker')
    finished_jobs = sagemaker_client.list_training_jobs()['TrainingJobSummaries']
    prefix = f'fahr-{tag.replace("/", "-").replace("_", "-")}'
    previous_jobs = [j for j in finished_jobs if j['TrainingJobName'].startswith(prefix)]
    return f'{prefix}-{len(previous_jobs)}' if previous_jobs else None


def get_next_job_name(tag):
    """
    Given the training run tag, determine the appropriate name for the next job run.
    SageMaker only.
    """
    previous_job_name = get_previous_job_name(tag)

    if previous_job_name:
        previous_job_num = int(previous_job_name[-1])
        next_job_num = previous_job_num + 1
        return f'{previous_job_name[:-1]}{str(next_job_num)}'
    else:
        return f'fahr-{tag}-1'


def validate_path(path):
    """
    Checks that a path exists and is a directory.
    """
    path = pathlib.Path(path)
    if not path.exists():
        raise ValueError("Output parameter must point to an existing directory.")
    if not path.is_dir():
        raise ValueError("Output path must be a directory.")
    return path


__all__ = ['TrainJob', 'fetch', 'copy_resources']
