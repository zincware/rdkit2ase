import ase
from rdkit import Chem

from rdkit2ase.rdkit2ase import ase2rdkit
from rdkit2ase.utils import find_connected_components
from ase.build import separate


def match_substructure(
    atoms: ase.Atoms, pattern: str | Chem.Mol | ase.Atoms
) -> tuple[tuple[int, ...]]:
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


def get_substructure(
    atoms: ase.Atoms, pattern: str | Chem.Mol | ase.Atoms
) -> list[ase.Atoms]:
    return [atoms[match] for match in match_substructure(atoms, pattern)]


def iter_fragments(atoms: ase.Atoms) -> list[ase.Atoms]:
    """Iterate over the molecules in an ASE Atoms object."""
    if "connectivity" in atoms.info:
        # connectivity is a list of tuples (i, j, bond_type)
        connectivity = atoms.info["connectivity"]
        for component in find_connected_components(connectivity):
            yield atoms[list(component)]
    else:
        for molecule in separate(atoms):
            yield molecule
