==========
User Guide
==========

Assuming you've read the `Quickstart`_, you now know how to run remote training jobs using the ``fahr`` command line. This section covers additional features and usage patterns useful for working with this tool.

.. _Quickstart: https://residentmario.github.io/fahr/quickstart.html

Motivation
----------

`Machine learning is an experimental discipline <https://www.residentmar.io/2019/02/23/data-science-experiments.html>`_, so it's not obvious what an performant model will look like right away. You have to experiment with a lot of ideas (and potentially fumble with many of them) to get there.

Your earliest experiments should be simple baseline models that you can hopefully run locally. However, once you start digging into the model&mdash;more layers, larger inputs, more epochs, hyperparameter tuning&mdash;you will get to the point where the model can no longer be comfortably run on your local machine.

Once a long training job is running on your machine, you can't really do anything else there because the training job is using up most of your compute resources.  It's a better user experience if you can bundle the training job up, ship it off to some remote service somewhere, and get back to iterating on your ideas. You won't know the result of the experiment until some number of minutes or hours later&mdash;but that shouldn't stop you from concurrently trying out other ideas as well while you wait for that one to finish.

``fahr`` lets you do that.

The workflow
------------

I use the following simple but effective workflow. `Also demonstrated here <https://github.com/ResidentMario/progressive-resizing/blob/master/notebooks/build-models.ipynb>`_. Note that this example uses the `SageMaker driver`_, but all drivers are equally compatible.

This example uses AWS SageMaker, and requires you have configured your environment to work with AWS SageMaker (see the `Sagemaker driver`_ section), but other drivers are similar.

.. _SageMaker driver: https://residentmario.github.io/fahr/drivers/sagemaker.html

1. Create a ``models`` folder in your repo.
2. Create a folder for your first remote experiment, e.g. ``resnet48_128``.
3. ``pip freeze > requirements.txt`` or ``conda env export --file environment.yml`` to create a package dependencies manifest.
4. Copy your training notebook (e.g. ``model-dev.ipynb``) or script (e.g. ``train.py``) to the folder. You may need to make some minor code adjustments to make it compatible with your training driver.
5. Run ``fahr init`` to initialize the job assets:

    .. code-block:: bash

        $ fahr init train.py \
            --train-driver='sagemaker' \
            --train-image='default-gpu' \
            --config.output_path=s3://your-s3-bucket/path/ \
            --config.role_name=your-s3-role

   Note: if you are in an environment lacking a GPU, use ``--build-driver='default-cpu'``.

   You should now have two model training artifacts in your folder, a ``Dockerfile`` and a ``run.sh`` entrypoint, which you may optionally edit.
6. Run ``fahr fit`` and launch the job:

    .. code-block:: bash

        $ fahr fit train.py \
            --train-driver='sagemaker' \
            --build-driver='local-gpu' \
            --config.output_path=s3://your-s3-bucket/path/ \
            --config.role_name=your-s3-role

You will get a message to ``stdout`` once the job is running explaining where to go to track it.

For jobs after the first one, you can use the handy ``copy`` method to move any files you need over:

    .. code-block:: bash

        $ fahr copy ../resnet48_128/ ./ \
            --include-training-artifact='train.py'

One of the great things about this workflow is that it makes it easy to brute-force model hyperparameter search. For example, to find the optimal ``batch_size`` parameter for the model in the demo repo I didn't bother writing additional code to perform a grid search&mdash;I just launched three jobs concurrently, one each at 512, 256, and 128 samples per batch.
