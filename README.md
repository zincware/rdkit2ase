[![zincware](https://img.shields.io/badge/Powered%20by-zincware-darkcyan)](https://github.com/zincware)
[![codecov](https://codecov.io/gh/zincware/rdkit2ase/graph/badge.svg?token=Q0VIN03185)](https://codecov.io/gh/zincware/rdkit2ase)
[![PyPI version](https://badge.fury.io/py/rdkit2ase.svg)](https://badge.fury.io/py/rdkit2ase)

# rdkit2ase - Interface between the rdkit and ASE package.

Installation via `pip install rdkit2ase`.

```py
from rdkit2ase import rdkit2ase, ase2rdkit

atoms: ase.Atoms = rdkit2ase(mol)
mol = ase2rdkit(atoms)
```

```py
from rdkit2ase import smiles2atoms

atoms: ase.Atoms = smiles2atoms("O")

print(atoms)
>>> Atoms(symbols='OH2', pbc=False)
```

### Limitations

- `rdkit2ase.ase2rdkit` won't be able to detect higher order bonds.
