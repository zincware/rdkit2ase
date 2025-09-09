[![zincware](https://img.shields.io/badge/Powered%20by-zincware-darkcyan)](https://github.com/zincware)
[![codecov](https://codecov.io/gh/zincware/rdkit2ase/graph/badge.svg?token=Q0VIN03185)](https://codecov.io/gh/zincware/rdkit2ase)
[![PyPI version](https://badge.fury.io/py/rdkit2ase.svg)](https://badge.fury.io/py/rdkit2ase)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.15423477.svg)](https://doi.org/10.5281/zenodo.15423477)
[![Docs](https://github.com/zincware/rdkit2ase/actions/workflows/pages.yaml/badge.svg)](https://zincware.github.io/rdkit2ase/)
[![status](https://joss.theoj.org/papers/4c1ec2a0b66ed6baaf3fe93066696d5f/status.svg)](https://joss.theoj.org/papers/4c1ec2a0b66ed6baaf3fe93066696d5f)

# rdkit2ase - Interface between the rdkit and ASE package.

Installation via `pip install rdkit2ase`. For more information please visit the
[documentation](https://zincware.github.io/rdkit2ase/).

A common use case is to create 3D structures from SMILES strings. This can be
achieved using the `rdkit2ase.rdkit2ase` function.

```py
import ase
from rdkit import Chem
from rdkit2ase import rdkit2ase, ase2rdkit

mol = Chem.MolFromSmiles("O")
atoms: ase.Atoms = rdkit2ase(mol)
mol = ase2rdkit(atoms)
```

Because this is such a common use case, there is a convenience function
`rdkit2ase.smiles2atoms` that combines the two steps.

```py
import ase
from rdkit2ase import smiles2atoms

atoms: ase.Atoms = smiles2atoms("O")

print(atoms)
>>> Atoms(symbols='OH2', pbc=False)
```

## Packmol Interface

Given the molecular units, you can build periodic boxes with a given density
using the `rdkit2ase.pack` function.

If you have [packmol](https://github.com/m3g/packmol) (at least `v20.15.0`) you
can use the rdkit2ase interface as follows:

```py
from rdkit2ase import pack, smiles2conformers

water = smiles2conformers("O", 2)
ethanol = smiles2conformers("CCO", 5)
density = 1000  # kg/m^3
box = pack([water, ethanol], [7, 5], density)
print(box)
>>> Atoms(symbols='C10H44O12', pbc=True, cell=[8.4, 8.4, 8.4])
```

Many additional features are described in the
[documentation](https://zincware.github.io/rdkit2ase/).
