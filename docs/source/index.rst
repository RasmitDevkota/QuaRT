QuaRT Documentation
======================

QuaRT
-----

**QuaRT** is a free/open-source toolkit for the exploration of quantum methods for radiation transport, particularly for astrophysical and cosmological problems, developed by the Computational Cosmology group at Georgia Tech. It is distributed under the `MIT License`_.

.. _MIT License: https://en.wikipedia.org/wiki/MIT_License

Installation
-------------------------

QuaRT is still in the development stage, so the latest nightly version of QuaRT is available on the GitHub under the `main branch`_.

.. _main branch: https://github.com/RasmitDevkota/QuaRT.git

The latest stable release can be installed from the PyPI index as follows:

.. code:: bash

    python -m pip install virtualenv
    python -m virtualenv .venv
    .venv/bin/activate
    python -m pip install PyQuaRT

The source code for the latest stable release can be found under the latest ``stable/*`` branch.

The latest nightly version of QuaRT can be installed from source as follows:

.. code:: bash

    git clone https://github.com/RasmitDevkota/QuaRT.git
    cd QuaRT
    python -m pip install virtualenv
    python -m virtualenv .venv
    .venv/bin/activate
    python -m pip install -e .

Documentation
-------------------------------

Documentation is hosted online on ReadtheDocs at `https://pyquart.readthedocs.io <https://pyquart.readthedocs.io>`__.

Acknowledgements
----------------

This research was supported in part through research cyberinfrastructure resources and services provided by the Partnership for an Advanced Computing Environment (PACE) at the Georgia Institute of Technology, Atlanta, Georgia, USA [@PACE].

Contributing
--------------------

We welcome contributions from anyone! A GitHub Issue may be used for making requests or reporting bugs. Please read our `contributing guidelines`_.

.. _contributing guidelines: https://github.com/RasmitDevkota/QuaRT/blob/main/CONTRIBUTING.md

Contact and Feedback
--------------------

For bug reports and feature requests, please `file a GitHub Issue`_.

.. _file a GitHub Issue: https://github.com/RasmitDevkota/QuaRT/issues

If you have any questions, comments, or concerns regarding QuaRT, feel free to contact Rasmit Devkota at `rdevkota3@gatech.edu <mailto:rdevkota3@gatech.edu>`__.

.. toctree::

   Home <self>
   API Reference

