=========
SageMaker
=========

Overview
--------

`AWS SageMaker <https://aws.amazon.com/sagemaker/>`_ is Amazon AWS's fully managed machine learning infrastructure as a service offering. It bundles a hodgepodge of ML features including model training, hyperparameter search, model deployment, data labelling, and hosted Jupyter notebooks. However model training and hosted Jupyter notebooks are its most salient features.

Training a model via SageMaker requires also interacting with Amazon S3, which it uses for artifact storage, and Amazon ECR, which it uses for Docker image storage.

Specs
-----

SageMaker can run training jobs on a broad variety of instances types and sizes, which vary widely in capabilities and cost. The full list is available in the `SageMaker pricing guide <https://aws.amazon.com/sagemaker/pricing/>`_. Multi-machine (distributed) training is also possible.

Prerequisites
-------------

You will need to have an account in AWS (which also means having a credit card on file).

Your training artifact must write model artifact(s) to the ``/opt/ml/model`` folder at the end of its execution. SageMaker bundles files in this folder and uploads them to S3 at the end of its execution.

Your current IAM user must have permission to get and assume an execution role. That role must have the following permissions: SageMaker full access, ECR read-write access, S3 read-write access, EC2 run access.

The S3 output path parameter must point to an S3 path that your current IAM user has read access to.

Basic usage
-----------

TODO

Advanced configuration
----------------------

TODO