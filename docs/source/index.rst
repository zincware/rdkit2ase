rdkit2ase Documentation
=======================

rdkit2ase provides a bridge between RDKit_ and ASE_.
It further integrates features from Packmol_ and NetworkX_.

.. code-block:: python

   import rdkit2ase

   water = rdkit2ase.smiles2conformers("O", numConfs=10)
   etoh = rdkit2ase.smiles2conformers("CCO", numConfs=10)
   box = rdkit2ase.pack(
      data=[water, etoh], counts=[5, 5], density=800, packmol="packmol.jl"
   )

   mol = rdkit2ase.ase2rdkit(water[0])
   graph = rdkit2ase.ase2networkx(box)

More examples are provided below.

Installation
------------

You can install rdkit2ase from PyPI:

.. code-block:: console

   (.venv) $ pip install rdkit2ase

From Source
-----------

To install and develop rdkit2ase from source, we recommend using `uv <https://docs.astral.sh/uv>`_.

More information and installation instructions can be found in the `UV documentation <https://docs.astral.sh/uv/getting-started/installation/>`_.

.. code-block:: console

   (.venv) $ git clone https://github.com/zincware/rdkit2ase
   (.venv) $ cd rdkit2ase
   (.venv) $ uv sync
   (.venv) $ source .venv/bin/activate

Documentation
-------------

.. toctree::
   :maxdepth: 2

   ase_tools
   rdkit_tools
   packmol_tools
   networkx_tools
   modules


.. _RDKit: https://www.rdkit.org/
.. _ASE: https://wiki.fysik.dtu.dk/ase/
.. _Packmol: https://github.com/m3g/Packmol.jl
.. _NetworkX: https://networkx.org/
