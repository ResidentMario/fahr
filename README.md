# alekseylearn

A simple CLI tool for running my machine learning training jobs on cloud compute. Intended for personal use, but it may eventually grow in scope to be something generally useful.

## How it works

First, some lingo:

* **training artifact** &mdash; A file (either `.ipynb` or `.py`) which, when executed correctly, produces a model artifact, e.g. a model training script or notebook.
* **model artifact** &mdash; A file which defines a machine learning model, e.g. a neural weight matrix.

Generating a model artifact with `alekseylearn` requires doing the following:

1. Building a Docker image based on your training artifact (the folder containing it must also have a `requirements.txt` file defining dependencies).
2. Uploading that Docker image to a container registry.
3. Executing that Docker image, saving the resulting model artifact somewhere.
4. Downloading that model artifact to your local machine.

This process requires an image build driver, a model training driver, a cloud storage service, and a container registry.

However, you only need to worry about the image build driver and the model training driver, as all of the other services are dictated by your choice of model training driver.

Current model training drivers supported:

* `sagemaker` (AWS)

Planned:

* Kaggle Kernels
* Google ML Flow

Current image build drivers supported:

* `local` (your local machine)
* `local-gpu` (your local machine with GPU support)

There are driver-specific configurations that must be set that are documented in the sections that follow.

### `sagemaker`

[AWS SageMaker](https://aws.amazon.com/sagemaker/) is Amazon AWS's fully managed machine learning platform.

#### How it works

`alekseylearn` will run the training job by building a custom SageMaker-compatible Docker image locally, then uploading that to AWS ECR. Then, it will execute that artifact using the AWS SageMaker API. The resulting model artifact will be sent to Amazon S3 blob storage, from which you retrieve it when the job is done.

#### How to run it

To run a `sagemaker` CLI job:

```bash
$ alekseylearn fit $MODEL_ARTIFACT_FILEPATH \
    --build_driver='local' \
    --train_driver='sagemaker' \
    --config.output_path=$S3_ARTIFACT_DIRECTORY \
    --config.role_name=$EXECUTION_ROLE_NAME
$ alekseylearn fetch $LOCAL_TARGET_DIRECTORY \
    $MODEL_IMAGE_TAG \
    $S3_ARTIFACT_FILEPATH
```

Where:

* `MODEL_ARTIFACT_FILEPATH` is the a path to the file defining the model artifact.
* `S3_ARTIFACT_DIRECTORY` is the S3 directory the model artifact will be deposited in.
* `MODEL_IMAGE_TAG` is the Docker image tag associated with the model. It is dependent on the model filepath: e.g. if `MODEL_ARTIFACT_FILEPATH=.../foo/model.py` then `MODEL_IMAGE_TAG=foo/model`.
* `EXECUTION_ROLE_NAME` is the nice name of a user role with the necessary permissions (if you are unsure what this means see the second bullet in "Configuration").

#### Prerequisites

* Your training artifact must write model artifact(s) to the `/opt/ml/model` folder at the end of its execution.
* Your current IAM user must have permission to get and assume the `EXECUTION_ROLE_NAME`. That role must have the following permissions: SageMaker full access, ECR read-write access, S3 read-write access, EC2 run access.
* `S3_ARTIFACT_DIRECTORY` must point to a path that your current IAM user has read access to.

#### Further configuration

Training will be performed on one `ml.c4.2xlarge` instance by default, which provides 8 cores, 15 GB of RAM, and no GPU on a 2.9 GHz Intel Xeon. At time of writing this costs $0.557/hour.

You can configure what kind and number of compute instance you will use by passing `--config.train_instance_count=$COUNT` and `--config.train_instance_type=$TYPE` to `fit`. For a list of options see the [SageMaker pricing guide](https://aws.amazon.com/sagemaker/pricing/).

### `kaggle`

Coming soon!

### `ml-engine`

Coming soon!