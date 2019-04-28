==========
Quickstart
==========

This page is a brisk instruction to using ``fahr``.

First install ``fahr``, if you haven't done so already:

.. code-block:: bash

    $ pip install fahr[kaggle]

For the purposes of this demo we will use a very simple ``train.py`` which builds a tiny ``keras`` feedforward network on synthetic data. This script uses ``numpy`` and ``keras`` so we'll need to install these packages also:

.. code-block:: bash

    $ yes | pip install numpy keras
    $ curl https://gist.githubusercontent.com/ResidentMario/ab3ce65a9bd5bc4061237a374fd9e67a/raw/train.py -o train.py

Executing this file saves a model definition file to your local disk:

.. code-block:: bash

    $ python train.py
    $ ls
    model.h5        train.py

``fahr`` takes a **training artifact** such as this one as input. Training artifacts may be either ``.py`` Python scripts or ``.ipynb`` Jupyter notebooks. It generally builds a Docker container wrapping this artifact in a way which is compatible with your chosen **training driver**, then uses the training driver's native API to upload the container and kick off a training job based on that container.

Let's step through that process now, using this file as our training artifact and `Kaggle Kernels <https://www.kaggle.com/kernels>`_ as our training driver.

You will need to have access to the Kaggle API to continue. You will need to have a Kaggle account, so create one if you haven't done so already. Then, `follow Kaggle's instructions <https://github.com/Kaggle/kaggle-api#api-credentials>`_ to download an auth token to the correct location on your local machine.

Once everything is tidy you will be able to run the following in your terminal (in all of the lines of code that follow, replace ``$USERNAME`` with your username on Kaggle):

.. code-block:: bash

    $ fahr fit 'train.py' --train-driver='kaggle' --config.username='$USERNAME'
    fahr.fahr - INFO - Writing kernel metadata to "$PATH/kernel-metadata.json".
    fahr.fahr - INFO - Fitting $USERNAME/train training job.
    fahr.fahr - INFO - The training job is now running. To track training progress visit https://www.kaggle.com/$USERNAME/train. To download finished model artifacts run fahr fetch --driver="kaggle" ./ "$USERNAME/train" after training is complete.

You can visit the web interface to your training job by clicking on the kernel link (`here's mine <https://www.kaggle.com/residentmario/train?scriptVersionId=13462676>`_). Once the job has completed running, you can go ahead and download a **model artifact** using the command documented in the log.

.. code-block:: bash

    fahr fetch --train-driver="kaggle" ./ "$USERNAME/train"

This will download ``model.h5`` to your local machine.

And that's it, you're done!

Every training driver in ``fahr`` uses the same API pattern: ``fit`` to turn a training artifact into a model artifact "on the cloud", then ``fetch`` to download it locally. However, the configuration options (specified via ``--config.$OPTION_NAME``) vary between training drivers. The ``kaggle`` driver is extremely simple: the only required additional argument is ``--config.username``. Other drivers require more arguments and more advance configuration, but are also more powerful. To learn about the drivers available in ``fahr``, or to see more options for working with ``kaggle``, check out the requisite sections of the docs.