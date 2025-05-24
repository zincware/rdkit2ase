import ase
from rdkit import Chem
from rdkit2ase.rdkit2ase import ase2rdkit


def match_substructure(atoms: ase.Atoms, pattern: str|Chem.Mol|ase.Atoms) -> tuple[tuple[int, ...]]:
    if isinstance(pattern, str):
        # assume smiles or smarts and convert to RDKit Mol
        pattern = Chem.MolFromSmarts(pattern)
    elif isinstance(pattern, ase.Atoms):
        pattern = ase2rdkit(pattern)
    elif not isinstance(pattern, Chem.Mol):
        raise TypeError("Pattern must be a string, ase.Atoms, or Chem.Mol")

    mol = ase2rdkit(atoms)
    matches = mol.GetSubstructMatches(pattern)
    return matches

def get_substructure(atoms: ase.Atoms, pattern: str|Chem.Mol|ase.Atoms) -> list[ase.Atoms]:
    return [atoms[match] for match in match_substructure(atoms, pattern)]
