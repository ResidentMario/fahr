from io import StringIO
import base64
import pathlib

import docker
import jinja2

class TrainJob:
    def __init__(self, filepath, driver='sagemaker', overwrite=False, config=None):
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
        overwrite: bool
            If set to False `TrainJob` will respect any Dockerfile and run.sh files already present
            in the build directory. If set to True it will overwrite them.
        config: dict
            A dict of driver-specific configuration variables used to inform how the job is run.
            The precise list of configuration options differs from driver to driver. See the
            documentation for more information.
        """
        filepath = pathlib.Path(filepath)
        dirpath = pathlib.Path(filepath).parent
        if not filepath.exists():
            raise ValueError("The training artifact points to a non-existent file.")
        if filepath.suffix != '.ipynb':
            raise NotImplementedError("Currently only Jupyter notebooks are supported.")
        if driver != 'sagemaker':
            raise NotImplementedError("Currently only AWS SageMaker is supported.")

        # TODO: allow conda code requirement definitions as well
        reqfile = dirpath / 'requirements.txt'
        if not reqfile.exists():
            raise ValueError("No requirements.txt file present in the build directory.")

        dockerfile = dirpath / 'Dockerfile'
        if not dockerfile.exists() or overwrite:
            create_dockerfile('sagemaker', dirpath, filepath)

        runfile = dirpath / 'run.sh'
        if not runfile.exists() or overwrite:
            create_runfile('sagemaker', dirpath, filepath)

        self.dirpath = dirpath
        self.filepath = filepath
        self.dockerfile = dockerfile
        self.runfile = runfile
        self.driver = driver
        self.config = config if config is not None else dict()

        self.docker_client = docker.client.from_env()
        self.tag = f'{self.dirpath.stem}/{self.filepath.stem}'

    def build(self):
        """
        Builds the model training image locally.
        """
        self.docker_client.images.build(path=self.dirpath.as_posix(), tag=self.tag, rm=True)

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

            account_id = sts_client.get_caller_identity()['Account']
            region = session.region_name
            repository=f'{account_id}.dkr.ecr.{region}.amazonaws.com/{self.tag}'

            # TODO: try to catch the botocore.errorfactory.RepositoryNotFound error
            try:
                ecr_client.list_images(registryId=account_id, repositoryName=self.tag)
            except:
                ecr_client.create_repository(repositoryName=self.tag)

            token = ecr_client.get_authorization_token(registryIds=[account_id])
            encoded_auth = token['authorizationData'][0]['authorizationToken']
            username, password = base64.b64decode(encoded_auth).decode().split(':')
            registry = token['authorizationData'][0]['proxyEndpoint']

            self.docker_client.login(username, password, registry=registry)
            image.tag(repository, 'latest')
            self.docker_client.images.push(repository)

            self.session = session
            self.repository = repository
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

            # FIXME: temporarily using this role name for testing using the Quilt org
            default_role_name = 'aleksey_sagemaker_role'
            role_name = self.config.pop('role_name', default_role_name)
            # TODO: try to catch the botocore.errorfactory.RepositoryNotFound error
            try:
                role = iam_client.get_role(RoleName=role_name)
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

            # FIXME: figure out the AWS session authorization song and dance
            sts_client = self.sts_client if hasattr(self, 'sts_client') else boto3.client('sts')
            assumed_role = sts_client.assume_role(
                RoleArn=role['Role']['Arn'],
                RoleSessionName='alekseylearn_test_session'
            )
            session = sage.Session()
            exec_role = sage.get_execution_role()

            # if `push` was previous called the repository will already have been set
            # otherwise we must build the repository again here
            # TODO: use a lazy-loaded attribute
            if not hasattr(self, 'repository'):
                account_id = session.boto_session.client('sts').get_caller_identity()['Account']
                region = session.boto_session.region_name
                # TODO: extract this to a helper method as it is used in push also
                self.repository = f'{account_id}.dkr.ecr.{region}.amazonaws.com/{self.tag}'

            clf = sage.estimator.Estimator(
                self.repository, exec_role, 1, 'ml.c4.2xlarge', 
                output_path="s3://alpha-quilt-storage/aleksey/alekseylearn-test",
                sagemaker_session=self.session
            )
            clf.fit()
        else:
            raise NotImplementedError

    def extract(self, path):
        """
        Extracts the model artifacts generated by the training job to `path`.

        Parameters
        ----------
        path: str or pathlib.Path
            Directory to write the model artifact to.
        """
        path = validate_path(path)

        if self.driver == 'sagemaker':
            # TODO: implement
            raise NotImplementedError
        else:
            raise NotImplementedError

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
        self.extract(path)


def create_dockerfile(driver, dirpath, filepath):
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
        # TODO: make this a common component of create_dockerfile and create_runfile via helper.
        template_env = jinja2.Environment(loader=jinja2.PackageLoader('alekseylearn', 'templates'))
        template_text = template_env.get_template('sagemaker/Dockerfile').render(
            filepath=filepath.name
        )
        with open(dirpath / 'Dockerfile', 'w') as f:
            f.write(template_text)
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
            template_env = jinja2.Environment(
                loader=jinja2.PackageLoader('alekseylearn', 'templates')
            )
            template_text = template_env.get_template('sagemaker/run.sh').render(
                run_cmd=run_cmd
            )
            with open(dirpath / 'run.sh', 'w') as f:
                f.write(template_text)
        else:
            raise NotImplementedError
    else:
        raise NotImplementedError


def validate_path(path):
    path = pathlib.Path(path)
    if not path.exists:
        raise ValueError("Output parameter must point to an existing directory.")
    if not path.is_dir():
        raise ValueError("Output path must be a directory.")
    return path

__all__ = ['TrainJob']
