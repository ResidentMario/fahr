# alekseylearn

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

All training jobs launched via `alekseylearn` require three things:

* A path to a training artifact
* An output path that the model artifact will be written to (and may then be fetched from).
* Either a `requirements.txt` or `environment.yml` file in the directory containing the model artifact; or a `--envfile=$PATH_TO_ENVFILE` argument pointing at a compatible environment file.

Besides that, there are driver-specific configurations that must be set that are documented in the sections that follow.

### `sagemaker`

[AWS SageMaker](https://aws.amazon.com/sagemaker/) is Amazon AWS's fully managed machine learning platform.

To run a `sagemaker` CLI job:

```bash
$ alekseylearn fit $MODEL_FILEPATH --config.output_path=$S3_ARTIFACT_FILEPATH
$ alekseylearn fetch $S3_ARTIFACT_FILEPATH
```

`alekseylearn` allows you to run a training job by uploading your training artifact as a custom SageMaker-compatible Docker image to AWS ECR (the AWS container registry offering), then executing that artifact using the AWS SageMaker API (which executes on an Amazon EC2 and saves the resulting model artifact to Amazon S3 blob storage). To successfully execute a job using the `sagemaker` driver ensure the following:

* The training artifact writes the model artifact(s) to the `/opt/ml/model` folder at the end of its execution.
* `config` is parameterized with an S3 `output_path` (e.g. `'s3://bucket/subpath'`) that your current IAM user has read access to.
* Your `output_path` includes the word "sagemaker" (this is for compatibility with the default roles SageMaker creates, which are scoped to only allow putting objects containing this fragment).
* Your current IAM user has permission to get and assume a SageMaker-compatible role (see the next section).
* There is either an `alekseylearn_sagemaker_role` [IAM role](https://docs.aws.amazon.com/IAM/latest/UserGuide/id_roles.html) with the necessary permissions present in your account, or you pass a `'role_name'` to `config` parameterized with a role with the necessary permissions. SageMaker will [assume this role](https://docs.aws.amazon.com/IAM/latest/UserGuide/id_roles_create_for-service.html) to run your training job. Permissions required: SageMaker full access, ECR read-write access, S3 read-write access, EC2 run access. Note that the [default SageMaker-created role](https://docs.aws.amazon.com/sagemaker/latest/dg/automatic-model-tuning-ex-role.html) is insufficient.

Training will be performed on one `ml.c4.2xlarge` instance by default, which provides 8 cores, 15 GB of RAM, and no GPU on a 2.9 GHz Intel Xeon. At time of writing this costs $0.557/hour. You can configure your own compute by passing `train_instance_count` and `train_instance_type` to `config`; for a list of alternative options see the [SageMaker pricing guide](https://aws.amazon.com/sagemaker/pricing/).-

### `kaggle`

Coming soon!

### `ml-engine`

Coming soon!