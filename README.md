# fahr [![docs passing](https://img.shields.io/badge/docs-passing-green.svg?style=flat-square)](https://residentmario.github.io/fahr/index.html)

`fahr` is a command-line tool for building machine learning models on
cloud hardware with as little overhead as possible.

`fahr` provides a simple unified interface to model training services like AWS SageMaker and Kaggle Kernels. By offloading model training to the cloud, `fahr` aims to make machine learning experimentation easy and fast.

## How it works

First, some lingo:

* **training artifact** &mdash; A file (either `.ipynb` or `.py`) which, when executed correctly, produces a model artifact, e.g. a model training script or notebook.
* **model artifact** &mdash; A file which defines a machine learning model, e.g. a neural weight matrix.

`fahr` turns a training artifact into a model artifact, using the magic of the cloud. Or, specifically, by:

1. Building a Docker image based on your training artifact and uploading it to a container registry.
2. Executing that Docker image, saving the resulting model artifact somewhere.
3. Downloading that model artifact to your local machine.

The current model training drivers supported are:

* `sagemaker` (AWS SageMaker)
* `kaggle` (Kaggle Kernels)

To learn more about `fahr` [check out the docs](https://residentmario.github.io/fahr/index.html).