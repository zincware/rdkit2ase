[![zincware](https://img.shields.io/badge/Powered%20by-zincware-darkcyan)](https://github.com/zincware)
# rdkit2ase - Interface between the rdkit and ASE package.

Installation via `pip install rdkit2ase`.

```py
from rdkit2ase import rdkit2ase, ase2rdkit

atoms: ase.Atoms = rdkit2ase(mol)
mol = ase2rdkit(atoms)
```

### Limitations
- `rdkit2ase.ase2rdkit` won't be able to detect higher order bonds.
