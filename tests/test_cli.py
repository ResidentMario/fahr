import unittest
import pytest
from click.testing import CliRunner
import pathlib
import os
import shutil
from unittest.mock import patch, Mock

from fahr import cli as fahr_cli
SAGEMAKER_FILEPATH = pathlib.Path('./fixtures/train_sagemaker/train.py').expanduser()\
    .absolute().as_posix()
KAGGLE_FILEPATH = pathlib.Path('./fixtures/train_kaggle/train.py').expanduser()\
    .absolute().as_posix()


class TestInit(unittest.TestCase):
    def setUp(self):
        self.cli = CliRunner()

    # TODO: investigate why this fails on Travis CI/CD
    @pytest.mark.xfail
    def test_sagemaker_init(self):
        with self.cli.isolated_filesystem():
            cwd = os.getcwd()
            shutil.copy(SAGEMAKER_FILEPATH, cwd.rstrip('/') + '/train.py')
            args = [
                'train.py',
                '--train-driver=sagemaker',
                '--config.output_path=s3://nonexistent-bucket/',
                '--config.role_name=fahr-test-role'
            ]
            assert self.cli.invoke(fahr_cli.init, args).exit_code == 0
            assert os.path.exists(cwd.rstrip('/') + '/Dockerfile')
            assert os.path.exists(cwd.rstrip('/') + '/run.sh')

    def test_kaggle_init(self):
        with self.cli.isolated_filesystem():
            cwd = os.getcwd()
            shutil.copy(KAGGLE_FILEPATH, cwd.rstrip('/') + '/train.py')
            args = [
                'train.py',
                '--train-driver=kaggle',
                '--config.username=TestUser'
            ]
            assert self.cli.invoke(fahr_cli.init, args).exit_code == 0
            assert os.path.exists(cwd.rstrip('/') + '/kernel-metadata.json')


def test_copy():
    cli = CliRunner()
    with cli.isolated_filesystem():
        cwd = os.getcwd()
        args = [
            pathlib.Path(SAGEMAKER_FILEPATH).parent.as_posix(),
            cwd,
            '--include-training-artifact=train.py'
        ]
        assert cli.invoke(fahr_cli.copy, args).exit_code == 0
        assert os.path.exists(cwd.rstrip('/') + '/Dockerfile')
        assert os.path.exists(cwd.rstrip('/') + '/run.sh')
        assert os.path.exists(cwd.rstrip('/') + '/train.py')


class TestFetch(unittest.TestCase):
    def setUp(self):
        self.cli = CliRunner()

    def test_kaggle_fetch(self):
        with patch('fahr.cli.TrainJob') as train_job_mock:
            args = [
                './',
                'TestUser/train',
                '--train-driver=kaggle'
            ]
            assert self.cli.invoke(fahr_cli.fetch, args).exit_code == 0
            train_job_mock.assert_called_once_with(
                job_name='TestUser/train', train_driver='kaggle'
            )
            train_job_mock.return_value.fetch.assert_called_once_with(
                './', extract=True
            )

    def test_sagemaker_fetch(self):
        with patch('fahr.cli.TrainJob') as train_job_mock:
            args = [
                './',
                'train-sagemaker-train-1',
                '--train-driver=sagemaker'
            ]
            assert self.cli.invoke(fahr_cli.fetch, args).exit_code == 0
            train_job_mock.assert_called_once_with(
                job_name='train-sagemaker-train-1', train_driver='sagemaker'
            )
            train_job_mock.return_value.fetch.assert_called_once_with(
                './', extract=True
            )


class TestFit(unittest.TestCase):
    def setUp(self):
        self.cli = CliRunner()

    def test_kaggle_fit(self):
        with self.cli.isolated_filesystem():
            cwd = os.getcwd()
            shutil.copy(KAGGLE_FILEPATH, cwd.rstrip('/') + '/train.py')
            with patch('fahr.cli.TrainJob') as train_job_mock:
                args = [
                    'train.py',
                    '--train-driver=kaggle',
                    '--config.username=TestUser'
                ]
                assert self.cli.invoke(fahr_cli.fit, args).exit_code == 0
                train_job_mock.assert_called_once_with(
                    build_driver='local', config={'username': 'TestUser'},
                    filepath='train.py', overwrite=False, train_driver='kaggle'
                )
                train_job_mock.return_value.fit.assert_called_once_with()

    def test_sagemaker_fit(self):
        with self.cli.isolated_filesystem():
            cwd = os.getcwd()
            shutil.copy(SAGEMAKER_FILEPATH, cwd.rstrip('/') + '/train.py')
            with patch('fahr.cli.TrainJob') as train_job_mock:
                args = [
                    'train.py',
                    '--train-driver=sagemaker',
                    '--config.role=TestRole',
                    '--config.output_path=s3://nonexistent-bucket/'
                ]
                assert self.cli.invoke(fahr_cli.fit, args).exit_code == 0
                train_job_mock.assert_called_once_with(
                    build_driver='local', filepath='train.py', overwrite=False,
                    train_driver='sagemaker',
                    config={'role': 'TestRole', 'output_path': 's3://nonexistent-bucket/'}
                )
                train_job_mock.return_value.fit.assert_called_once_with()
