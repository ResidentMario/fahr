import unittest
from unittest.mock import patch, call, Mock, ANY
import pytest
import pathlib
from copy import deepcopy
import base64

import boto3
import botocore
from botocore import UNSIGNED
from botocore.client import Config
from botocore.stub import Stubber, ANY

import fahr
from fahr import TrainJob

FILEPATH = pathlib.Path('./fixtures/train_simple/train.py').expanduser().absolute()
ENVFILE = pathlib.Path('./fixtures/train_simple/requirements.txt').expanduser().absolute()
kaggle_kwargs = {
    'build_driver': 'local', 'train_driver': 'kaggle',
    'config': {'username': 'TestUsername'}
}
sagemaker_kwargs = {
    'build_driver': 'local', 'train_driver': 'sagemaker',
    'config': {'output_path': 's3://nonexistent-bucket/out/', 'role_name': 'TestRole'}
}
def create_resources_mock(*args, **kwargs):
    return (
        pathlib.Path('./fixtures/train_simple/Dockerfile').expanduser().absolute(),
        pathlib.Path('./fixtures/train_simple/run.sh').expanduser().absolute()
    )

class TestTrainJobInit(unittest.TestCase):
    """
    Test TrainJob object initialization. The only work that is performed at this step is a lot of
    parameter validation.
    """
    def test_invalid_train_driver(self):
        with pytest.raises(NotImplementedError):  # train driver not in list of accepted drivers
            TrainJob(FILEPATH, train_driver='nondriver')

    def test_invalid_build_driver(self):
        with pytest.raises(NotImplementedError):  # build driver not in list of accepted drivers
            TrainJob(FILEPATH, **{**kaggle_kwargs, **{'build_driver': 'nondriver'}})

    def test_kaggle_init_valid(self):
        # test basic initialization with the kaggle driver
        with patch('fahr.fahr.create_kaggle_resources'):
            j = TrainJob(FILEPATH, **kaggle_kwargs)

        assert j.filepath == FILEPATH
        assert j.train_driver == 'kaggle'
        assert j.tag == 'TestUsername/train'

    def test_kaggle_init_invalid_no_username(self):
        kwargs = deepcopy(kaggle_kwargs)
        del kwargs['config']['username']
        with pytest.raises(ValueError):  # no username
            TrainJob(FILEPATH, **kwargs)
        
    def test_kaggle_init_invalid_non_existant_model_file(self):
        with pytest.raises(ValueError):  # non-existent model definition file
            TrainJob('./not/a/file.py', **kaggle_kwargs)

    @patch('pathlib.Path.exists', lambda _: True)
    def test_kaggle_init_invalid_model_file_fails_regex(self):
        with pytest.raises(ValueError):  # file name fails kernel regex
            TrainJob('!@#$.py', **kaggle_kwargs)

    def test_kaggle_init_invalid_sources(self):
        # sources are not valid input types
        kwargs = deepcopy(kaggle_kwargs)
        kwargs['config'].update({'dataset_sources': {}})
        with pytest.raises(ValueError):
            TrainJob(FILEPATH, **kwargs)

        kwargs = deepcopy(kaggle_kwargs)
        kwargs['config'].update({'kernel_sources': {}})
        with pytest.raises(ValueError):
            TrainJob(FILEPATH, **kwargs)

        kwargs = deepcopy(kaggle_kwargs)
        kwargs['config'].update({'competition_sources': {}})
        with pytest.raises(ValueError):
            TrainJob(FILEPATH, **kwargs)

    def test_kaggle_init_invalid_envfile(self):
        with pytest.raises(ValueError):  # an envfile is not allowed with this driver
            TrainJob(FILEPATH, envfile=ENVFILE, **kaggle_kwargs)

    # TODO: test the case that envfile is unspecified and the default envfile doesn't exist

    @patch('pathlib.Path.exists', lambda _: True)
    def test_init_invalid_envfile(self):
        with pytest.raises(ValueError):  # envfile is not a requirements.txt
            TrainJob(FILEPATH, envfile='./environment.yml', **sagemaker_kwargs)

    def test_init_invalid_no_envfile(self):
        with pytest.raises(ValueError):  # envfile doesn't exist
            TrainJob(FILEPATH, envfile='./not/a/requirements.txt', **sagemaker_kwargs)

    def test_sagemaker_init_valid(self):
        # test basic initialization with the sagemaker driver
        with patch('fahr.fahr.create_resources', new=create_resources_mock):
            j = TrainJob(FILEPATH, **sagemaker_kwargs)

        assert j.filepath == FILEPATH
        assert j.train_driver == 'sagemaker'
        assert j.tag == 'train-simple-train'

    def test_sagemaker_init_invalid_no_output_path(self):
        kwargs = deepcopy(sagemaker_kwargs)
        del kwargs['config']['output_path']

        with pytest.raises(ValueError):  # no output_path
            TrainJob(FILEPATH, **kwargs)

    def test_sagemaker_init_invalid_non_s3_output_path(self):
        kwargs = deepcopy(sagemaker_kwargs)
        kwargs['config']['output_path'] = 'not-s3://nonexistent-bucket/out/'

        with pytest.raises(ValueError):  # invalid output_path
            TrainJob(FILEPATH, **kwargs)

    def test_sagemaker_init_invalid_no_role_name(self):
        kwargs = deepcopy(sagemaker_kwargs)
        del kwargs['config']['role_name']

        with pytest.raises(ValueError):  # no role_name
            TrainJob(FILEPATH, **kwargs)


class TestTrainJobBuild(unittest.TestCase):
    """
    Test the TrainJob build method, used to build the model training image.
    """
    def test_kaggle_build(self):  # no-op
        with patch('fahr.fahr.create_kaggle_resources'):
            j = TrainJob(FILEPATH, **kaggle_kwargs)
    
        j.docker_client = Mock()
        j.build()

        assert j.docker_client.images.build.called == False
    
    def test_sagemaker_build(self):
        with patch('fahr.fahr.create_resources', new=create_resources_mock):
            j = TrainJob(FILEPATH, **sagemaker_kwargs)
        j.docker_client = Mock()
        j.build()

        j.docker_client.images.build.assert_called_once_with(
            path=str(FILEPATH.parent),
            rm=True,
            tag='train-simple-train'
        )


def test_kaggle_push():  # no-op
    with patch('fahr.fahr.create_kaggle_resources'):
        j = TrainJob(FILEPATH, **kaggle_kwargs)
    j.docker_client = Mock()
    j.push()
    assert j.docker_client.images.push.called == False


class TestTrainJobSagemakerPush(unittest.TestCase):
    """
    Test the TrainJob push method with the SageMaker API. This requires substituting out
    STS and ECR clients using the botocore stubber utility and writing our own authorization
    API bodies.
    """
    def setUp(self):
        sts = boto3.client(
            'sts', region_name='us-east-1', config=Config(signature_version=UNSIGNED)
        )
        ecr = boto3.client(
            'ecr', region_name='us-east-1', config=Config(signature_version=UNSIGNED)
        )
        self.sts_stubber = Stubber(sts)
        self.ecr_stubber = Stubber(ecr)

        self.ecr_stubber.add_response(
            'list_images', service_response={},
            expected_params={'registryId': '1234', 'repositoryName': 'train-simple-train'}
        )
        self.ecr_stubber.add_response(
            'get_authorization_token', service_response={
                    'authorizationData': [
                        {'authorizationToken': base64.b64encode(b'abcd:1234').decode('utf-8'),
                        'proxyEndpoint': 'https://1234.dkr.ecr.region.amazonaws.com'}
                    ]
            },
            expected_params={'registryIds': ['1234']}
        )
        self.sts_stubber.add_response(
            'get_caller_identity', service_response={'Account': '1234'}, expected_params=None
        )
        self.ecr_stubber.activate()
        self.sts_stubber.activate()

        def client(svc):
            """
            Replace the new service client constructor with a constructor that returns the stubbed
            versions.
            """
            if svc == 'sts':
                return sts
            elif svc == 'ecr':
                return ecr
            else:
                raise ValueError
        self.client = client
    
    def tearDown(self):
        self.ecr_stubber.deactivate()
        self.sts_stubber.deactivate()

    def test_sagemaker_push(self):
        with patch('fahr.fahr.create_resources', new=create_resources_mock), \
            patch('boto3.session.Session') as sesh_mock, \
            patch('boto3.client', new=self.client), \
            patch('docker.client.from_env'):
            sesh_mock.return_value.region_name = 'us-east-1'

            j = TrainJob(FILEPATH, **sagemaker_kwargs)
            j.push()

            # The actual test itself checks that:
            # 1. The username and password for the Docker login were retrieved from the AWS auth
            #    API correctly.
            # 2. The ECR registry associated with the user ID is correctly deduced.
            # 3. The image is pushed to the right ECR URL.
            j.docker_client.login.assert_called_once_with(
                'abcd', '1234', registry='https://1234.dkr.ecr.region.amazonaws.com'
            )
            j.docker_client.images.push.assert_called_once_with(
                '1234.dkr.ecr.us-east-1.amazonaws.com/train-simple-train'
            )


class TestTrainJobSagemakerTrain(unittest.TestCase):
    """
    Test the TrainJob train method with the SageMaker API. This requires substituting out
    STS and IAM clients using the botocore stubber utility and writing our own authorization
    API bodies, and stubbing out of the SageMaker and job name finder APIs.
    """
    def setUp(self):
        sts = boto3.client(
            'sts', region_name='us-east-1', config=Config(signature_version=UNSIGNED)
        )
        iam = boto3.client(
            'iam', region_name='us-east-1', config=Config(signature_version=UNSIGNED)
        )
        self.sts_stubber = Stubber(sts)
        self.iam_stubber = Stubber(iam)

        self.iam_stubber.add_response(
            'get_role', service_response={'Role': {
                    'Path': '/',
                    'RoleName': '"TestRole',
                    'RoleId': 'AIDIODR4TAW7CSEXAMPLE',
                    'Arn': 'arn:aws:iam::1234:role/TestRole',
                    'CreateDate': '2013-04-18T05:01:58Z'
                }
            },
            expected_params={'RoleName': 'TestRole'}
        )
        self.sts_stubber.add_response(
            'get_caller_identity',
            service_response={'Arn': 'arn:aws:sts::1234:assumed-role/TestRole'},
            expected_params=None
        )
        self.sts_stubber.add_response(
            'get_caller_identity', service_response={'Account': '1234'}, expected_params=None
        )
        self.iam_stubber.activate()
        self.sts_stubber.activate()

        def client(svc):
            """
            Replace the new service client constructor with a constructor that returns the stubbed
            versions.
            """
            if svc == 'sts':
                return sts
            elif svc == 'iam':
                return iam
            else:
                raise ValueError
        self.client = client
    
    def tearDown(self):
        self.sts_stubber.deactivate()
        self.iam_stubber.deactivate()

    # This method is unusually slow for unclear reasons, but seemingly *not* due to any hidden
    # network requests: running this test with `--disable-socket` (via the `pytest-socket` plugin)
    # does not expedite the process or result in any errors.
    def test_sagemaker_train(self):
        with patch('fahr.fahr.create_resources', new=create_resources_mock), \
            patch('boto3.client', new=self.client), \
            patch('sagemaker.get_execution_role',
                  return_value='arn:aws:sts::1234:assumed-role/TestRole'), \
            patch('fahr.fahr.get_next_job_name', return_value='fahr-train-simple-train-1'), \
            patch('sagemaker.estimator.Estimator') as estimator_mock:
            j = TrainJob(FILEPATH, **sagemaker_kwargs)
            j.session = Mock()
            j.session.get_credentials.return_value.get_frozen_credentials.return_value =\
                botocore.credentials.ReadOnlyCredentials(
                    access_key='1234', secret_key='1234', token=None
                )
            j.session.region_name = 'us-east-1'
            j.repository = '1234.dkr.ecr.us-east-1.amazonaws.com/train-simple-train'

            j.train()

            # The actual test: the sagemaker API's estimator object gets called correctly.
            estimator_mock.assert_called_once_with(
                image_name='1234.dkr.ecr.us-east-1.amazonaws.com/train-simple-train',
                role='arn:aws:sts::1234:assumed-role/TestRole',
                train_instance_count=1,
                train_instance_type='ml.c4.2xlarge',
                output_path='s3://nonexistent-bucket/out/',
                sagemaker_session=ANY
            )
            estimator_mock.return_value.fit.assert_called_once_with(
                job_name='fahr-train-simple-train-1', wait=False
            )


class TestTrainJobKaggleTrain(unittest.TestCase):
    """
    Test the TrainJob train method with the Kaggle API.
    """
    def test_kaggle_train(self):
        with patch('subprocess.run') as run_mock, \
            patch('fahr.fahr.create_kaggle_resources'):
            j = TrainJob(FILEPATH, **kaggle_kwargs)
            j.fit()
            run_mock.assert_called_once_with(
                ["kaggle", "kernels", "push", "-p", FILEPATH.parent.as_posix()],
                stderr=ANY, stdout=ANY
            )


# TODO: refactor fetch to be a TrainJob method, and TrainJob init to allow for old job init.
# TODO: and properly test the fetch method itself, for both SageMaker and Kaggle APIs.
def test_train_job_fetch():
    """
    Test the TrainJob object fetch method (which calls out to the fahr method).
    """
    with patch('fahr.fahr.fetch') as fetch_mock, \
        patch('fahr.fahr.create_kaggle_resources'), \
        patch('fahr.fahr.create_resources', new=create_resources_mock):
            j = TrainJob(FILEPATH, **kaggle_kwargs)
            j.job_name = 'fahr-train-simple-train-1'
            j.fetch('./')
            fetch_mock.assert_called_once_with(
                './', 'TestUsername/train', extract=True, job_name='fahr-train-simple-train-1',
                remote_path=None, train_driver='kaggle'
            )
            fetch_mock.reset_mock()

            j = TrainJob(FILEPATH, **sagemaker_kwargs)
            j.job_name = 'fahr-train-simple-train-1'
            j.fetch('./')
            fetch_mock.assert_called_once_with(
                './', 'train-simple-train', extract=True, job_name='fahr-train-simple-train-1',
                remote_path='s3://nonexistent-bucket/out/', train_driver='sagemaker'
            )


class TestCopyResources(unittest.TestCase):
    """
    Test the copy_resources utility method.
    """
    def test_invalid_no_dockerfile(self):
        with patch('shutil.copy'), \
            patch('pathlib.Path.exists', lambda p: p != FILEPATH.parent / 'Dockerfile'):
            with pytest.raises(ValueError):
                fahr.copy_resources(FILEPATH.parent, './')

    def test_invalid_no_runfile(self):
        with patch('shutil.copy'), \
            patch('pathlib.Path.exists', lambda p: p != FILEPATH.parent / 'run.sh'):
            with pytest.raises(ValueError):
                fahr.copy_resources(FILEPATH.parent, './')

    def test_invalid_no_requirements_file(self):
        with patch('shutil.copy'), \
            patch('pathlib.Path.exists', lambda p: p != FILEPATH.parent / 'requirements.txt'):
            with pytest.raises(ValueError):
                fahr.copy_resources(FILEPATH.parent, './')

    def test_invalid_no_model_definition_file(self):
        with patch('shutil.copy'), \
            patch('pathlib.Path.exists', lambda p: p != FILEPATH.parent / 'train.py'):
            with pytest.raises(ValueError):
                fahr.copy_resources(FILEPATH.parent, './', training_artifact='train.py')

    def test_valid_copy(self):
        with patch('shutil.copy') as copy_mock:
            fahr.copy_resources(FILEPATH.parent, './', training_artifact='train.py')

        for fn in ['Dockerfile', 'train.py', 'requirements.txt', 'run.sh']:
            in_path = (FILEPATH.parent / fn).as_posix()
            assert call(in_path, fn) in copy_mock.call_args_list


def test_get_repository_info():
    """
    Test the method for retrieving repository information based on account and tag from AWS.
    """
    sts = boto3.client('sts', region_name='us-east-1', config=Config(signature_version=UNSIGNED))
    sts_stubber = Stubber(sts)
    sts_stubber.add_response(
        'get_caller_identity', service_response={'Account': '1234'}, expected_params=None
    )

    with sts_stubber, patch('boto3.session.Session') as sesh_mock:
        sesh_mock.region_name = 'us-east-1'

        region, account_id, repository =\
            fahr.fahr.get_repository_info(sesh_mock, 'train-simple-train', sts)

        assert repository == '1234.dkr.ecr.us-east-1.amazonaws.com/train-simple-train'
        assert region == 'us-east-1'
        assert account_id == '1234'


class TestGetPreviousJobName(unittest.TestCase):
    """
    Test the helper method for retrieving the previous SageMaker job name.
    """
    def setUp(self):
        sagemaker = boto3.client(
            'sagemaker', region_name='us-east-1', config=Config(signature_version=UNSIGNED)
        )
        self.sagemaker_stubber = Stubber(sagemaker)
        self.sagemaker_stubber.add_response(
            'list_training_jobs', expected_params={},
            service_response={
                'NextToken': 'string',
                'TrainingJobSummaries': [ 
                    {
                        'CreationTime': 1234,
                        'LastModifiedTime': 1234,
                        'TrainingEndTime': 1234,
                        'TrainingJobArn': 'arn',
                        'TrainingJobName': 'fahr-train-simple-train-1',
                        'TrainingJobStatus': 'Completed'
                    },
                    {
                        'CreationTime': 1234,
                        'LastModifiedTime': 1234,
                        'TrainingEndTime': 1234,
                        'TrainingJobArn': 'arn',
                        'TrainingJobName': 'fahr-unrelated-train-job-1',
                        'TrainingJobStatus': 'Completed'
                    },
                    {
                        'CreationTime': 1234,
                        'LastModifiedTime': 1234,
                        'TrainingEndTime': 1234,
                        'TrainingJobArn': 'arn',
                        'TrainingJobName': 'fahr-train-simple-train-2',
                        'TrainingJobStatus': 'Completed'
                    }
                ]
            }
        )
        self.sagemaker_stubber.activate()

        def client(svc):
            return sagemaker
        self.client = client

    def test_get_previous_job_name(self):
        expected = 'fahr-train-simple-train-2'
        with patch('boto3.client', new=self.client):
            result = fahr.fahr.get_previous_job_name('train-simple-train')
        
        assert expected == result

    def tearDown(self):
        self.sagemaker_stubber.deactivate()


def test_get_next_job_name():
    """
    Test the helper method for retrieving the next SageMaker job name.
    """
    expected = 'fahr-train-simple-train-1'
    with patch('fahr.fahr.get_previous_job_name', return_value=None):
        result = fahr.fahr.get_next_job_name('train-simple-train')
    assert expected == result
    
    expected = 'fahr-train-simple-train-2'
    with patch('fahr.fahr.get_previous_job_name', return_value='fahr-train-simple-train-1'):
        result = fahr.fahr.get_next_job_name('train-simple-train')
    assert expected == result
