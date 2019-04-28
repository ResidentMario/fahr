==========
User Guide
==========

Assuming you've read the `Quickstart`_, you now know how to run remote training jobs using the ``fahr`` command line. This section covers additional features and usage patterns useful for working with this tool.

.. _Quickstart: https://residentmario.github.io/fahr/quickstart.html

Motivation
----------

The `progressive resizing <https://github.com/ResidentMario/progressive-resizing>`_ repo demonstrates a good pattern for using ``fahr``, and it's the example I will use throughout this section of the documentation.

`Machine learning is an experimental discipline <https://www.residentmar.io/2019/02/23/data-science-experiments.html>`_, so it's not obvious what an performant model will look like right away. You have to experiment with a lot of ideas (and potentially fumble with many of them) to get there.

Your earliest experiments should be simple baseline models that you can hopefully run locally. However, once you start digging into the model - more layers, more complex loss functions, longer training cycles, advanced hyperparameter tuning - the complexity of the model will start to exceed your ability to train it locally without long waiting times.

Using a powerful cloud Jupyter environment like AWS SageMaker Notebooks will buy you time, but with a sufficiently complex model you *will* get the to the point where your model takes an hour or more to train. And once a long training job is running in your notebook, you can't really do anything else there because it's waiting for your training job to finish. If you start working on your models in a separate notebook that not only breaks up your flow but also, if you're still on the same instance, gets in the way of your training job, which is (hopefully) using up most of your compute resources.

It's a better user experience if you can bundle up your current training job, ship it off to some remote service somewhere, and get back to iterating on your ideas. You won't know whether the particular idea you're experimenting with is "good" or not until that hour passes and you get back a result, but that shouldn't stop you from concurrently trying out other ideas as well while you wait for that one to finish.

``fahr`` lets you do that.

The workflow
------------

The ``build-models.ipynb`` notebook in the demo repo demonstrates the workflow. `Open it yourself <https://github.com/ResidentMario/progressive-resizing/blob/master/notebooks/build-models.ipynb>`_ to follow along. Note that this example uses the `SageMaker driver`_, but all drivers are equally compatible.

.. _SageMaker driver: https://residentmario.github.io/fahr/drivers/sagemaker.html


The first time you want to run a remote training job you will need to create some resources:

1. Once your models get too complex to train comfortably locally, create a ``models`` folder in your repo.
2. Create a folder for your first remotely-trained experiment: in ``progressive-resizing`` this was a model I named ``resnet48_128``.
3. Add a ``requirements.txt`` file to the folder that documents the packages required for your model traning jobs to run (``keras``, ``numpy``, etcetera).
4. Create a model training artifact: a ``train.py`` or ``train.ipynb`` file that defines the training regimen.
5. Go into the model folder and run ``fahr init``:

    .. code-block:: bash

        $ fahr init train.py \
            --train-driver='sagemaker' \
            --build-driver='local-gpu' \
            --config.output_path=$S3_ARTIFACT_DIRECTORY \
            --config.role_name=$EXECUTION_ROLE_NAME

   Note the ``--build-driver='local-gpu'`` argument. This specifies that you are building a image that uses the GPU, and requires that you train in a GPU-equipped and configured environment. See the `SageMaker driver`_ instructions to learn more.
6. You should now have some model training artifacts in your folder. In all non-``kaggle`` cases this will be a ``Dockerfile`` compatible with your training driver and a ``run.sh`` script that acts as the entrypoint to that image. You can edit these to your liking. For exampe, in the demo repo I added some commands to download the dataset to ``run.sh`` (`see here <https://github.com/ResidentMario/progressive-resizing/blob/master/resnet48_128/run.sh>`_).
7. Run ``fahr fit`` and launch the job:

    .. code-block:: bash

        $ fahr fit train.py \
            --train-driver='sagemaker' \
            --build-driver='local-gpu' \
            --config.output_path=$S3_ARTIFACT_DIRECTORY \
            --config.role_name=$EXECUTION_ROLE_NAME

The process for every run after the first is similar, but simpler:

1. Create a new folder for the experiment.
2. Copy the files over:

    .. code-block:: bash

        $ fahr copy ../resnet48_128/ ./ \
            --include-training-artifact='train.py'

3. Edit as necessary - add or remote packages from the ``requirements.txt``, update environment variables in the ``Dockerfile``, etcetera.
4. Run ``fahr fit`` again to launch the new job.

One of the great things about this workflow is that it makes it easy to brute-force model hyperparameter search. When I was trying to find the optimal ``batch_size`` in the demo repo, I didn't bother writing additional code to perform a grid search, I just launched three jobs concurrently, one each at 512, 256, and 128 samples per batch.
