.. Polnet documentation master file, created by
   sphinx-quickstart on Wed Feb 25 16:22:30 2026.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

PolNet documentation
====================

**PolNet** is a Python package for generating synthetic datasets of the
cellular context for Cryo-Electron Tomography (Cryo-ET).

The simulation pipeline is split into two main stages:

1. **Sample simulation** — generates a synthetic 3-D volume of interest
   (VOI) populated with membranes, filaments, cytosolic proteins, and
   membrane-bound proteins.
2. **Microscope simulation & tomogram reconstruction** — applies a TEM
   acquisition model (tilt-series projection, CTF, noise) and
   reconstructs the final tomogram using IMOD tools.

All simulation parameters are controlled through a single YAML
configuration file.  The ``polnet`` CLI is the main entry point.

Quick start
-----------

.. code-block:: console

   pip install .
   polnet config/all_features.yaml -v

See the `README <https://github.com/anmartinezs/polnet>`_ for full
installation and usage instructions.

API Reference
-------------

.. toctree::
   :maxdepth: 3
   :caption: Contents:

   modules


Indices and tables
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
