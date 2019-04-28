==========
Quickstart
==========

This page is a brisk instruction to using ``fahr``.

Walkthrough
-----------

First install ``fahr``, if you haven't done so already:

.. code-block:: bash

    pip install git+https://github.com/ResidentMario/fahr.git@master`

For the purposes of this demo, I've defined a very simple script, ``train.py``, which builds a tiny ``keras`` feedforward network on synthetic data. This script uses ``numpy`` and ``keras`` so we'll need to install these packages also:

.. code-block:: bash

    yes | pip install numpy keras
    curl https://gist.githubusercontent.com/ResidentMario/ab3ce65a9bd5bc4061237a374fd9e67a/raw/train.py -o train.py

Executing this file saves a model definition file to your local disk:

.. code-block:: bash

    python train.py
    ls
    model.h5        train.py

``fahr`` takes an **training artifact** such as this one as input. It builds a Docker container wrapping this artifact in a way which is compatible with your chosen **training driver**, then uses the training driver's native API to upload the container and kick off a training job based on that container.

Let's step through that process now, using this file as our training artifact and `Kaggle Kernels <https://www.kaggle.com/kernels>`_ as our training driver.

You will need to have access to the Kaggle API to continue. You will need to have a Kaggle account, so create one if you haven't done so already. Then, `follow Kaggle's instructions <https://github.com/Kaggle/kaggle-api#api-credentials>`_ to download an auth token to the correct location on your local machine.

Once everything is tidy you will be able to run the following in your terminal (in all of the lines of code that follow, replace ``$USERNAME`` with your username on Kaggle):

.. code-block:: bash

    fahr fit 'train.py' --train-driver='kaggle' --config.username='$USERNAME'

``fahr`` will log its status to the terminal as it process your job:

.. code-block:: bash

    fahr.fahr - INFO - Writing kernel metadata to "$PATH/kernel-metadata.json".
    fahr.fahr - INFO - Fitting $USERNAME/train training job.
    fahr.fahr - INFO - The training job is now running. To track training progress visit https://www.kaggle.com/$USERNAME/train. To download finished model artifacts run fahr fetch --driver="kaggle" ./ "$USERNAME/train" after training is complete.

You can visit the web interface to your training job by clicking on the kernel link. Once the job has completed running, you can go ahead and download a **model artifact** using the command documented in the log.

.. code-block:: bash

    fahr fetch --driver="kaggle" ./ "$USERNAME/train"

This will download ``model.h5`` to your local machine.

And that's it, you're done!