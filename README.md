[![zincware](https://img.shields.io/badge/Powered%20by-zincware-darkcyan)](https://github.com/zincware)
[![codecov](https://codecov.io/gh/zincware/rdkit2ase/graph/badge.svg?token=Q0VIN03185)](https://codecov.io/gh/zincware/rdkit2ase)
[![PyPI version](https://badge.fury.io/py/rdkit2ase.svg)](https://badge.fury.io/py/rdkit2ase)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.15423477.svg)](https://doi.org/10.5281/zenodo.15423477)
[![Docs](https://github.com/zincware/rdkit2ase/actions/workflows/pages.yaml/badge.svg)](https://zincware.github.io/rdkit2ase/)

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

## Packmol Interface

If you have [packmol](https://github.com/m3g/packmol) (at least `v20.15.0`) you
can use the rdkit2ase interface.

```py
from rdkit2ase import pack, smiles2conformers

water = smiles2conformers("O", 2)
ethanol = smiles2conformers("CCO", 5)
density = 1000  # kg/m^3
box = pack([water, ethanol], [7, 5], density)
print(box)
>>> Atoms(symbols='C10H44O12', pbc=True, cell=[8.4, 8.4, 8.4])
```

### Limitations

- `rdkit2ase.ase2rdkit` won't be able to detect higher order bonds.
