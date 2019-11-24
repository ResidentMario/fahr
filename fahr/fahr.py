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
import warnings

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
        self, filepath=None, job_name=None, train_image='default-cpu', train_driver='sagemaker',
        envfile=None, overwrite=False, config=None
    ):
        """
        Object encapsulating a machine learning training job.

        Parameters
        ----------
        filepath: str
            Path to the model definition to be executed remotely. A model definition
            is an executable code object (currently either a Jupyter notebook or a Python script,
            plus optional supporting files), whose output is a model artifact (a serialized model,
            a saved weights matrix, etcetera).

            One of `filepath` or `job_name` is required. If `filepath` is provided, a `TrainJob`
            instance with the `status` `"unlaunched"` is initialized.
        job_name: str
            If this parameter is provided, `TrainJob` will be initialized with this model training
            job, adopting its `status` and model artifact output location. The job can then be used
            to `fetch` this job's model artifacts.

            One of `filepath` or `job_name` is required. See `job_name` under "Properties" for more
            information.
        train_image: str
            This parameter controls the base image used for the Docker image in which training will
            be performed.
            
            The model definition (see `filepath`) is packaged into a Docker image before being sent
            off to train, creating a model training image. This operation is performed on your
            local machine (in the future, a `train_driver` parameter may allow for performing this
            step on remote compute). The following hard-coded images are provided:

            * 'default-cpu' -- Builds the model container on your local machine without GPU, using
              the `python:latest` Docker image.
            * 'default-gpu' -- Builds the model container on your local machine with GPU support,
               using the `tensorflow/tensorflow:1.10.1-gpu-py3` Docker image. Note that this option
               requires a machine with a GPU onboard and nvidia-docker
               (https://github.com/NVIDIA/nvidia-docker) installed.

            Alternatively, to specify a different base image, provide that image's tag as input.

            Note that no `train_image` is necessary when using the Kaggle job runner, as Kaggle
            does not provide the ability to adjust the runtime environment via the API.
        train_driver: str
            The training driver (service) is the compute platform that performs the training
            (executes the model training image). Currently the following options are supported:

            * 'sagemaker' -- Launches a model training job on Amazon using AWS SageMaker.
            * 'kaggle' -- Launches a model training job on Kaggle using Kaggle Kernels.

            Support for GCP ML Engine ('ml-engine') is planned.
        envfile: str, optional
            Optional path to a `requirements.txt` file. If left unspecified the file present in
            the build directory will be used (and if no `requirements.txt` is present, an error
            will be raised).
            
            The envfile defines the list of packages that will be included into the model training
            artifact, and should contain every dependency necessary for the model to train
            successfully.
            
            Support for the `conda` `environment.yaml` format is planned.
        overwrite: bool, default False
            Set this value to True to overwrite any already-created model definition files
            (e.g. `run.sh`).
        config: dict
            A dict of driver-specific configuration variables used to inform how the job is run.
            The precise list of configuration options differs from driver to driver. See the
            User Reference (https://residentmario.github.io/fahr/index.html) for more information.
        
        Properties
        ----------
        tag: str
            The tag associated with this job. For example, a `TrainJob` initialized with the
            `32_16_cnn.py` artifact in the `imagenet` folder might be given the `tag`
            `imagenet-32-16-cnn`.
        job_name: str
            The job's name. This property will not be available until `fit` has been called.
            The job name is closely linked to the tag, e.g. the above.
        """
        if job_name is None and filepath is None:
            raise ValueError('One of "job_name" or "filepath" is required.')
        if job_name is not None and filepath is not None:
            raise ValueError('Cannot specify both "job_name" and "filepath".')
        elif job_name is not None:
            self.train_image = train_image
            self.train_driver = train_driver
            self.job_name = job_name

            self._latest_status = 'submitted'
            self.status()
        else:  # filepath is not None
            filepath = pathlib.Path(filepath).absolute()
            dirpath = filepath.parent
            if not filepath.exists():
                raise ValueError('The training artifact points to a non-existent file.')
            if filepath.suffix != '.ipynb' and filepath.suffix != '.py':
                raise NotImplementedError(
                    'Currently only Jupyter notebooks and Python scripts are supported.'
                )
            if train_image != 'default-cpu' and train_image != 'default-cpu':
                logger.info(f'Using the {train_image!r} image as the base environment image.')
                # TODO: verify that the train_image is valid.
                # TODO: pull the image if necessary.
                # TODO: update the default image based on which training service is used.
            elif train_image == 'default-cpu':
                logger.info(f'Using the default CPU image as the base environment image.')
            elif train_image == 'default-gpu':
                logger.info(f'Using the default GPU image as the base environment image.')

            # Check driver-specific configuration requirements
            if train_driver == 'sagemaker':
                if (config is None or
                    'output_path' not in config or
                    not config['output_path'].startswith('s3://')):
                    raise ValueError('The SageMaker driver requires an output_path to "s3://".')

                if 'role_name' not in config:
                    raise ValueError('The SageMaker driver requires a role name.')

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
                filename = f'{filepath.stem}'.replace("/", "-").replace("_", "-")
                regex = '^[a-zA-Z0-9]{5,}'
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

            if envfile is None:
                default_envfile = dirpath / 'requirements.txt'
                if not default_envfile.exists():
                    logger.info(
                        f'No "requirements.txt" included in the model definition directory. '
                        f'Using the default environment.')
                    envfile = None
                else:
                    logger.info(f'Using "{default_envfile}" as the environment definition file.')
                    envfile = default_envfile
            else:
                envfile = pathlib.Path(envfile)
                if not envfile.exists():
                    raise ValueError(
                        f'Cannot set "envfile" to non-existent file {envfile.as_posix()}.'
                    )
                if not envfile.name == 'requirements.txt':
                    raise ValueError(
                        'Currently only environment definition files of the "requirements.txt" '
                        'type are supported.'
                    )
                logger.info(f'Using "{envfile}" as the environment definition file.')
                envfile = envfile.absolute().relative_to(pathlib.Path.cwd()).as_posix()

            if train_driver == 'sagemaker':
                dockerfile, runfile = create_sagemaker_resources(
                    train_image, train_driver, dirpath, filepath, envfile, overwrite
                )
            else:  # train_driver == 'kaggle':
                # Kaggle kernels run in a default environment. The web UI allows you to 
                # specify custom packages but the API does not currently support this feature.
                if envfile is not None:
                    logger.warning(
                        'An environment definition file is provided, but the Kaggle training '
                        'driver is currently in use. Kaggle does not currently support running '
                        'code in custom containers in the API. The default environment will be '
                        'used instead.'
                    )
                dockerfile, runfile = create_kaggle_resources(
                    dirpath, filepath, tag,
                    title=config.pop(
                        'title', filename.replace('_', ' ').replace('-', ' ').title()
                    ),
                    is_private=config.pop('is_private', False),
                    enable_gpu=config.pop('enable_gpu', False),
                    enable_internet=config.pop('enable_internet', False),
                    dataset_sources=config.pop('dataset_sources', []),
                    kernel_sources=config.pop('kernel_sources', []),
                    competition_sources=config.pop('competition_sources', []),
                )

            # The _latest_status internal prop tracks the most recent job status known by the
            # class. If the job is initialized from a local file, but not launched, it will be
            # set to 'unlaunched'. If the job has been launched, but never checked, it will be
            # 'submitted'. If the job has been launched, been checked, and the check shows that
            # the job is finished, it will be 'complete' or 'failed', depending on the job's
            # outcome.
            #
            # The status method is a public quasi-property (actually a method, as performing a
            # network request on a property is disingenous) which abstracts over this property.
            self._latest_status = 'unlaunched'

            self.dirpath = dirpath
            self.filepath = filepath
            self.train_image = train_image
            self.train_driver = train_driver
            self.config = config
            self.tag = tag
            self.dockerfile = dockerfile
            self.runfile = runfile
            self.docker_client = docker.client.from_env()

    def status(self):
        """
        Returns the status of this job instance. May be one of the following:
        * `unlaunched` --- This job has not been launched (sent to compute using `fit`) yet.
        * `submitted` --- This job has been submitted to cloud compute. It has not terminated yet.
        * `complete` --- This job has finished running, and you may call `fetch` to get your model.
        * `failed` --- This job has failed. Use the training driver's web tools to determine why.
        """
        if self._latest_status == 'unlaunched':
            return 'unlaunched'
        elif self._latest_status == 'submitted':
            if self.train_driver == 'sagemaker':
                job_info = get_sagemaker_job_info(self.job_name)
                online_status = job_info['TrainingJobStatus']
                # TODO: is an 'interrupted' status appropriate?
                if online_status in ['InProgress', 'Stopping', 'Stopped']:
                    return 'submitted'
                elif online_status == 'Completed':
                    self._latest_status = 'complete'
                    return 'complete'
                elif online_status == 'Failed':
                    return 'failed'
                else:
                    raise ValueError(
                        f'Checking the status of the SageMaker job returned unexpected status '
                        f'value {online_status}, indicating a possible schema change. Please '
                        f'file an issue on GitHub with a traceback.'
                    )
            else:  # self.train_driver == 'kaggle'
                online_status = subprocess.check_output(
                    ["kaggle", "kernels", "status", self.job_name]
                ).decode('utf-8')
                sentinel = f'{self.job_name} has status '
                if not sentinel in online_status:
                    raise ValueError(
                        f'Checking the status of the Kaggle job returned unexpected status '
                        f'value {online_status!r}, indicating a possible schema change. Please '
                        f'file an issue on GitHub with a traceback.'
                    )
                l = online_status.find(sentinel)
                l += len(sentinel) + 1
                r = online_status.find('"', l)
                online_status = online_status[l:r]

                if online_status == 'complete':
                    # Note that 'complete' only means that the job runner did not throw an error,
                    # e.g. every code cell ran or the script finished execution. The notebook or
                    # script could have itself failed; this is still reported as a 'complete' by
                    # the Kaggle API.
                    self._latest_status = 'complete'
                    return 'complete'
                elif online_status == 'running':
                    return 'submitted'
                elif online_status == 'failed':
                    return 'failed'
        else:  # self._latest_status in ['complete', 'failed']
            return self._latest_status

    @classmethod
    def from_model_definition(
        cls, filepath, train_image='default-cpu', train_driver='sagemaker',
        envfile=None, overwrite=False, config=None
    ):
        return cls(
            filepath=filepath, train_image=train_image, train_driver=train_driver,
            envfile=envfile, overwrite=overwrite, config=config
        )

    @classmethod
    def from_training_job(cls, job_name, train_driver='sagemaker', config=None):
        return cls(
            job_name=job_name, train_driver=train_driver, config=config
        )

    def build(self):
        """
        Builds the model training image. This is the first step in the model training workflow.
        """
        if self.train_driver != 'kaggle':
            if self.train_image == 'default-cpu' or self.train_image == 'default-gpu':
                path = self.dirpath.as_posix()
                logger.info(f'Building "{self.tag}" container image from "{path}".')
                self.docker_client.images.build(path=path, tag=self.tag, rm=True)
            else:
                raise NotImplementedError

    def push(self):
        """
        Pushes the model training image to a remote image repository. This is the second step in
        the model training workflow, as it is a necessary prerequesite to executing the model
        training image. The repository used depends on the `train_driver`:

        * 'sagemaker' -- Pushes the image to Amazon ECR.
        * 'kaggle' -- Does nothing, only the default runtime environment is available, so no
          registry is necessary.
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

        elif self.train_driver == 'kaggle':
            return

        else:
            raise NotImplementedError

    def train(self):
        """
        Launches a remote training job. The third and most critical step in the model
        training workflow. Where the job is launched depends on the `train_driver`:

        * sagemaker -- The job is run on an EC2 machine via the AWS SageMaker API.
        * kaggle -- The job is run in a Kaggle Kernel via the Kaggle API.
        """
        if self.train_driver == 'sagemaker':
            import boto3
            import sagemaker as sage

            iam_client = boto3.client('iam')
            sts_client = boto3.client('sts')

            # role_name is a required parameter at initialization time
            role_name = self.config['role_name']

            # TODO: try to catch the botocore.errorfactory.RepositoryNotFound error
            try:
                role_info = iam_client.get_role(RoleName=role_name)
            except:
                raise ValueError(
                    f'The {role_name} role does not exist, you must create it first.'
                )

            # if the current execution context is not the `role_name` role, assume it
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
                    auth['AccessKeyId'], auth['SecretAccessKey'], auth['SessionToken']
                )

            assumed_role_session = boto3.session.Session(
                aws_access_key_id=aws_access_key_id,
                aws_secret_access_key=aws_secret_access_key, 
                aws_session_token=aws_session_token
            )

            # If the repository identifier is not already known (from having run push beforehand)
            # relearn it here. We take care to use the boto3 Session object for this, not the
            # sagemaker Session object; the latter is slightly different.
            if not hasattr(self, 'repository'):
                _, _, self.repository = get_repository_info(
                    assumed_role_session, self.tag, sts_client
                )

            session = sage.Session(boto_session=assumed_role_session)
            execution_role = sage.get_execution_role(sagemaker_session=session)

            train_instance_count = self.config.pop('train_instance_count', 1)
            train_instance_type = self.config.pop('train_instance_type', 'ml.c4.2xlarge')
            output_path = self.config['output_path']
            clf = sage.estimator.Estimator(
                image_name=self.repository, role=execution_role, 
                train_instance_count=train_instance_count, train_instance_type=train_instance_type,
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
            self._latest_status = 'submitted'

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
            self.job_name = self.tag
            self._latest_status = 'submitted'

        else:
            raise NotImplementedError

    def fetch(self, local_path, extract=True):
        """
        Extracts the model artifacts generated by the training job to `path`. This is the last step
        in the model training workflow.

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
        ValueError -- Raised if you attempt to `fetch` without first running `fit`, or if the job
        is not finished running yet.
        """
        if self.status() == 'unlaunched':
            raise ValueError('Cannot fetch TrainJob model artifacts without fitting first.')
        
        # Validate path.
        local_path = pathlib.Path(local_path)
        if not local_path.exists():
            raise ValueError("'local_path' must point to an existing directory.")
        if not local_path.is_dir():
            raise ValueError("'local_path' must point to a directory.")

        if self.train_driver == 'sagemaker':
            job_info = get_sagemaker_job_info(self.job_name)
            remote_path = job_info['ModelArtifacts']['S3ModelArtifacts']

            remote_path_name_parts = remote_path.replace('s3://', '').split('/')
            bucket_name = remote_path_name_parts[0]
            # '1:' to exclude the bucket name part, and ':-3' to exclude the path fragment
            # SageMaker adds: /{job_name}/output/model.tar.gz, to get just the part in between
            bucket_path = '/'.join(remote_path_name_parts[1:-3])

            model_path = f'{bucket_path}/{self.job_name}/output/model.tar.gz'.lstrip('/')
            local_model_filepath = pathlib.Path(f'{local_path}/model.tar.gz').absolute()
            local_model_filepath_str = local_model_filepath.as_posix()
            local_model_directory_str = f'{local_model_filepath.parent.as_posix()}/'

            import boto3
            s3_client = boto3.client('s3')
            s3_client.download_file(bucket_name, model_path, local_model_filepath_str)

            if extract:
                tarfile.open(local_model_filepath_str).extractall()
                pathlib.Path(local_model_filepath_str).unlink()
                logger.info(f'Downloaded model artifact(s) to "{local_model_directory_str}".')
            else:
                logger.info(f'Downloaded model artifact(s) to "{local_model_filepath_str}".')

        elif self.train_driver == 'kaggle':
            subprocess.run(
                ["kaggle", "kernels", "output", self.job_name, "-p", local_path],
                stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT
            )
            logger.info(f'Downloaded model artifact(s) to "{local_path}".')

    def fit(self):
        """
        Executes a model training job, generating a model training artifact in a local repository.

        This is a convenience method that combines the first three steps of the model training
        workflow: building the model training image, pushing it to a remote repository, and
        executing it.
        
        Parameters
        ----------
        path: str or pathlib.Path
            Directory to write the model artifact to.
        """
        self.build()
        self.push()
        self.train()


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


def create_dockerfile(train_image, train_driver, dirpath, filepath, envfile):
    """
    Creates a Dockerfile compatible with the given drivers and writes to to disk.

    Parameters
    ----------
    train_image: str
        The train_image that will build the model image.
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
            filepath=filepath, envfile=envfile, train_image=train_image
        )
    else:
        raise NotImplementedError


def create_runfile(train_image, train_driver, dirpath, filepath):
    """
    Creates a run.sh entrypoint for the Docker image compatible with the given `train_driver`,
    and writes it to disk.

    Parameters
    ----------
    train_image: str
        The train_image (service) that will build the model image.
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


def create_sagemaker_resources(train_image, train_driver, dirpath, filepath, envfile, overwrite):
    """
    Creates file resources that will be used by this library in a target directory 
    (specifically, a Dockerfile and a run.sh entrypoint). Calls `create_runfile` and
    `create_dockerfile` as a subroutine.

    See also: create_kaggle_resources.

    Parameters
    ----------
    train_image: str
        The train_image (service) that will build the model image.
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
        create_dockerfile(train_image, train_driver, dirpath, filepath.name, envfile)
    else:  # dockerfile_exists
        logger.info(f'Creating new Dockerfile at "{dockerfile}".')
        create_dockerfile(train_image, train_driver, dirpath, filepath.name, envfile)

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
        create_runfile(train_image, train_driver, dirpath, filepath)
    else:  # runfile_exists
        logger.info(f'Creating new image entrypoint at "{runfile}".')
        create_runfile(train_image, train_driver, dirpath, filepath)  

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
    
    # The other create_*_resources methods return a (Dockerfile, runfile) tuple. Since Kaggle
    # does not make use of these assets, we return a (None, None) instead here.
    return None, None


def get_repository_info(session, tag, sts_client):
    """
    Identify the ECR repository for a given tag. SageMaker only.
    """
    account_id = sts_client.get_caller_identity()['Account']
    region = session.region_name

    repository = f'{account_id}.dkr.ecr.{region}.amazonaws.com/{tag}'
    return region, account_id, repository


def get_previous_jobs(tag):
    """
    Given the training run tag, determine the name of the job immediately previous to this
    one, assuming one exists. SageMaker only.
    """
    import boto3
    
    # TODO: support paginated requests, as otherwise most recent run can fall of the list
    sagemaker_client = boto3.client('sagemaker')
    finished_jobs = sagemaker_client.list_training_jobs()['TrainingJobSummaries']
    prefix = f'fahr-{tag.replace("/", "-").replace("_", "-")}'
    previous_jobs = [j for j in finished_jobs if j['TrainingJobName'].startswith(prefix)]
    return previous_jobs


def get_previous_job_name(tag):
    previous_jobs = get_previous_jobs(tag)
    prefix = f'fahr-{tag.replace("/", "-").replace("_", "-")}'
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


def get_sagemaker_job_info(job_name):
    import boto3
    sagemaker = boto3.client('sagemaker')
    return sagemaker.describe_training_job(TrainingJobName=job_name)


__all__ = ['TrainJob', 'copy_resources']
