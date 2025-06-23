rdkit2ase Documentation
=======================

The rdkit2ase package provides an interface between RDKit and ASE.
It further integrates packmol and networkx features and molecular mapping.

To install rdkit2ase, you can use pip:

.. code-block:: console

   (.venv) $ pip install mlipx


From Source
-----------

To install and develop rdkit2ase from source we recommend using :code:`https://docs.astral.sh/uv`.
More information and installation instructions can be found at https://docs.astral.sh/uv/getting-started/installation/ .

.. code:: console

   (.venv) $ git clone https://github.com/zincware/rdkit2ase
   (.venv) $ cd rdkit2ase
   (.venv) $ uv sync
   (.venv) $ source .venv/bin/activate

You can quickly switch between different :term:`MLIP` packages extras using :code:`uv sync` command.

Documentation
-------------

.. toctree::
   :maxdepth: 2

   ase_tools
   rdkit_tools
   packmol_tools
   networkx_tools
