.. fahr documentation master file, created by
   sphinx-quickstart on Sun Mar 24 20:50:36 2019.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Fahr: building ML models on the cloud
=====================================

``fahr`` is a command-line tool for building machine learning models on
cloud hardware with as little overhead as possible.

``fahr`` provides a simple unified interface to model training services like 
AWS SageMaker and Kaggle Kernels. By offloading model training to the cloud,
`fahr` aims to make machine learning experimentation easy and fast.

For a brief introduction refer to the `Quickstart`_.

To learn more about the motivations for developing ``fahr``, read the "Introducing fahr" article (forthcoming).

.. _Quickstart: https://residentmario.github.io/fahr/quickstart.html
.. _API Reference: https://residentmario.github.io/fahr/fahr.html


.. toctree::
   :maxdepth: 1

   installation.rst
   quickstart.rst
   User Guide <user_guide.rst>
   Kaggle Driver <drivers/kaggle.rst>
   SageMaker Driver <drivers/sagemaker.rst>
   API Reference <fahr.rst>

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
