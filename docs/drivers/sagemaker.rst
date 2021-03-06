=========
SageMaker
=========

Overview
--------

`AWS SageMaker <https://aws.amazon.com/sagemaker/>`_ is Amazon AWS's fully managed machine learning infrastructure as a service offering. It provides a hodgepodge of ML features, the most salient of which are its model training API and its hosted Jupyter notebooks interface.

Training a model via SageMaker requires also interacting with Amazon S3, which it uses for artifact storage, and Amazon ECR, which it uses for Docker image storage.

To learn more about this product see `"Building fully custom machine learning models on AWS SageMaker: a practical guide" <https://towardsdatascience.com/building-fully-custom-machine-learning-models-on-aws-sagemaker-a-practical-guide-c30df3895ef7>`_.

Specs
-----

SageMaker can run training jobs on a broad variety of instances types and sizes, which vary widely in capabilities and cost. The full list is available in the `SageMaker pricing guide <https://aws.amazon.com/sagemaker/pricing/>`_. Multi-machine (distributed) training is also possible.

By default training will be performed on a ``ml.c4.2xlarge`` instance, which provides 8 cores, 15 GB of RAM, and no GPU on a 2.9 GHz Intel Xeon. At time of writing this costs $0.557/hour. To learn how to specify alternative runtime environments see the section "Advanced configuration".

How it works
------------

You first build a Docker image defining your model environment locally, which gets uploaded to Amazon's container registry service, AWS ECR. SageMaker pulls this Docker image into an EC2 instance of your chosing configured with the SageMaker AMI, then runs the image. After execution terminates, it picks up any outputs written to the ``opt/ml/`` folder and compresses them to a tarfile on S3.

To download your model artifact, down load the file written to S3 and extract it.

Requirements
------------

You will need to have an account in AWS, and your account credentials must be available locally (`see the AWS docs for help with this <https://docs.aws.amazon.com/general/latest/gr/aws-sec-cred-types.html>`_).

You will need to have Docker installed and running. If your model requires a GPU, you must furthermore have an Nvidia GPU and the `nvidia docker <https://github.com/NVIDIA/nvidia-docker>`_ toolchain installed, and it must also use a container definition that has this toolchain installed. An easy way to get this environment is to build your image inside of a `SageMaker notebook <https://docs.aws.amazon.com/sagemaker/latest/dg/nbi.html>`_.

Your training artifact must write model artifact(s) to the ``/opt/ml/model`` folder during its execution. SageMaker targets these files for extraction. You probably don't want to do when you are running your training job locally. You can check the ``FAHR_EXECUTION_ENVIRONMENT`` environment variable to decide where to save your files:

.. code-block:: python

    import os
    # running remotely, e.g. launched via "fahr fit"
    if os.environ.get('FAHR_EXECUTION_ENVIRONMENT') == 'sagemaker':
        model.save('/opt/ml/model/model.h5')
    # running locally, e.g. launched via "python"
    else:
        model.save('model.h5')

Your current IAM user must have permission to get and assume an execution role with the following permissions: SageMaker full access, ECR read-write access, S3 read-write access, EC2 run access. If you do not have such a role you will need to create one.

The S3 output path parameter must point to an S3 path that your current IAM user has read access to.

Basic usage
-----------

To run a training job on AWS SageMaker:

.. code-block:: bash

    $ fahr fit $MODEL_ARTIFACT_FILEPATH \
        --train-driver='sagemaker' \
        --config.output_path=$S3_ARTIFACT_DIRECTORY \
        --config.role_name=$EXECUTION_ROLE_NAME

To download the model artifact after training is complete:

.. code-block:: bash

    $ fahr fetch $LOCAL_TARGET_DIRECTORY \
        $MODEL_IMAGE_TAG \
        $S3_ARTIFACT_FILEPATH

Where:

* ``$MODEL_ARTIFACT_FILEPATH`` is the a path to the file defining the model artifact.
* ``$S3_ARTIFACT_DIRECTORY`` is the S3 directory the model artifact will be deposited in.
* ``$MODEL_IMAGE_TAG`` is the Docker image tag associated with the model. It is dependent on the model filepath: e.g. if ``$MODEL_ARTIFACT_FILEPATH=.../foo/model.py`` then ``$MODEL_IMAGE_TAG=foo/model``.
* ``$EXECUTION_ROLE_NAME`` is the nice name of the role that SageMaker will assume whilst running this training job.

Limitations
-----------

The Amazon EC2 GPU instances that back SageMaker training runs provide CUDA version 9 and cuDNN version 7. When performing GPU training, the base image for your container must be one compatible with these two library versions. For Tensorflow for example this means that your image must be in the 1.5.0 through 1.12.0 version range (`source <https://www.tensorflow.org/install/source#tested_build_configurations>`_).

Advanced configuration
----------------------

You can configure what kind and number of compute instance you will use by passing ``--config.train_instance_count=$COUNT`` and ``--config.train_instance_type=$TYPE`` to ``fahr fit``. For a list of options see the `SageMaker pricing guide <https://aws.amazon.com/sagemaker/pricing/>`_.