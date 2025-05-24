import ase
from ase.build import separate
from rdkit import Chem

from rdkit2ase.rdkit2ase import ase2rdkit
from rdkit2ase.utils import find_connected_components


def match_substructure(
    atoms: ase.Atoms, pattern: str | Chem.Mol | ase.Atoms
) -> tuple[tuple[int, ...]]:
    """
    Find all matches of a substructure pattern in a given ASE Atoms object.

    Parameters
    ----------
    atoms : ase.Atoms
        The molecule or structure in which to search for substructure matches.
    pattern : str or Chem.Mol or ase.Atoms
        The substructure to search for, provided as a SMARTS/SMILES string,
        an RDKit Mol object, or an ASE Atoms object.

    Returns
    -------
    tuple of tuple of int
        A tuple of atom index tuples, each corresponding to one match of the pattern.
    """
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


def get_substructures(
    atoms: ase.Atoms, pattern: str | Chem.Mol | ase.Atoms
) -> list[ase.Atoms]:
    """
    Extract all matched substructures from an ASE Atoms object.

    Parameters
    ----------
    atoms : ase.Atoms
        The structure to search in.
    pattern : str or Chem.Mol or ase.Atoms
        The substructure pattern to match.

    Returns
    -------
    list of ase.Atoms
        List of substructure fragments matching the pattern.
    """
    return [atoms[match] for match in match_substructure(atoms, pattern)]


def iter_fragments(atoms: ase.Atoms) -> list[ase.Atoms]:
    """
    Iterate over connected molecular fragments in an ASE Atoms object.

    If a 'connectivity' field is present in `atoms.info`, it will be used
    to determine fragments. Otherwise, `ase.build.separate` will be used.

    Parameters
    ----------
    atoms : ase.Atoms
        A structure that may contain one or more molecular fragments.

    Yields
    ------
    ase.Atoms
        Each connected component (fragment) in the input structure.
    """
    if "connectivity" in atoms.info:
        # connectivity is a list of tuples (i, j, bond_type)
        connectivity = atoms.info["connectivity"]
        for component in find_connected_components(connectivity):
            yield atoms[list(component)]
    else:
        for molecule in separate(atoms):
            yield molecule
