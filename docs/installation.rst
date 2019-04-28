You can install ``fahr`` in the standard way using ``pip``:

.. code-block:: bash

    $ pip install fahr[all]


``fahr`` will not install training driver -specific libraries unprompted. To install everything pass an ``all`` selector as above. Alternatively you may download a specific subset:

.. code-block:: bash

    $ pip install fahr[sagemaker, kaggle]
