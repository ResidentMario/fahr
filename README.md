A simple utility package for running my machine learning training jobs on cloud compute. Intended for personal use, but it may eventually grow in scope to be something generally useful.

Current job runners supported:
* AWS SageMaker

Planned:
* Kaggle Kernels
* Google ML Flow

## Quickstart
To specify a training run, start by creating a `TrainJob` object pointed at a file of your choice defining the model training artifact. Currently this artifact must be a Jupyter notebook, but in the future I want to add support for Python scripts and shell scripts as well.

Successfully executing a job (via `TrainJob.run`) will create a model artifact in your local directory of choice. Additionally, a new `Dockerfile` and `run.sh` entrypoint script will be created in this directory, if one does not exist already.

You have a choice of `driver` which will perform the work. Successfully executing a job requires some further configuration which is driver-dependent. The sections below outline what you need to have or do.

### SageMaker
TODO: write this part.