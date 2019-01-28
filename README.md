A simple utility package for running my machine learning training jobs on cloud compute. Intended for personal use, but it may eventually grow in scope to be something generally useful.

Current job runners supported:
* AWS SageMaker

Planned:
* Kaggle Kernels
* Google ML Flow

## Quickstart
First, some lingo:
* **training artifact** &mdash; A file which, when executed correctly, produces a model artifact, e.g. a model training script or notebook.
* **model artifact** &mdash; A file which defines a machine learning model, e.g. a neural weight matrix.

To specify a training run, start by creating an `alekseylearn.TrainJob(path, driver)` object. `path` should point to a training artifact&mdash;currently this artifact must be a Jupyter notebook, but in the future I want to add support for Python scripts and shell scripts as well. `driver` should be the name of the driver you intend to run with (currently only `sagemaker` is allowed).

The directory containing the training artifact must also contain a `requirements.txt` file. This file will be used to `pip install` your dependendies inside of your container.

Then, execute `TrainJob.run(path)`, replacing `path` with the directory you'd like to write the model artifact(s) to.

Successfully executing the job will deposit the model artifact into your directory. Additionally, a new `Dockerfile` and `run.sh` entrypoint script will be created in this directory, if one does not exist already.

You have a choice of `driver` which will perform the work. Successfully executing a job requires some further configuration which is driver-dependent. The sections below outline what you need to have or do.

### `sagemaker`
[AWS SageMaker](https://aws.amazon.com/sagemaker/) is Amazon AWS's fully managed machine learning platform.

`alekseylearn` allows you to run a training job by uploading your training artifact as a custom SageMaker-compatible Docker image to AWS ECR (the AWS container registry offering), then executing that artifact using the AWS SageMaker API (which executes on an Amazon EC2 and saves the resulting model artifact to Amazon S3 blob storage).

To successfully execute a job using the `sagemaker` driver ensure the following:
* The training artifact writes the model artifact(s) to `/opt/ml/model`.
* `config` is parameterized with a `dict` with at least the `output_path` populated, e.g. `config={'output_path': 's3://bucket/subpath'}`.
* Your `output_path` includes the word "sagemaker" (this is for compatibility with the default roles SageMaker creates, which are scoped to only allow putting objects containing this fragment).
* Your current IAM user has read access to the S3 bucket and path specified in the `output_path`.
* Your current IAM user has permission to get and assume a SageMaker-compatible role (see the next section).
* There is either an `alekseylearn_sagemaker_role` [IAM role](https://docs.aws.amazon.com/IAM/latest/UserGuide/id_roles.html) with the necessary permissions present in your account, or you pass a `'role_name'` to `config` parameterized with a role with the necessary permissions. SageMaker will [assume this role](https://docs.aws.amazon.com/IAM/latest/UserGuide/id_roles_create_for-service.html) to run your training job. Permissions required: SageMaker full access, ECR read-write access, S3 read-write access, EC2 run access. Note that the [default SageMaker-created role](https://docs.aws.amazon.com/sagemaker/latest/dg/automatic-model-tuning-ex-role.html) is insufficient.

Note that training will be performed on one `ml.c4.2xlarge` instance by default, which provides 8 cores, 15 GB of RAM, and no GPU on a 2.9 GHz Intel Xeon. At time of writing this costs $0.557/hour. You can configure your own compute by passing `train_instance_count` and `train_instance_type` to `config`; for a list of alternative options see the [SageMaker pricing guide](https://aws.amazon.com/sagemaker/pricing/).

### `kaggle`
Coming soon!

### `ml-engine`
Coming soon!