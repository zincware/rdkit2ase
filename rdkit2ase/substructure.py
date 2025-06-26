import ase
from ase.build import separate
from rdkit import Chem

from rdkit2ase.ase2x import ase2rdkit
from rdkit2ase.utils import find_connected_components


def match_substructure(
    atoms: ase.Atoms,
    smiles: str | None = None,
    smarts: str | None = None,
    mol: Chem.Mol | None = None,
    fragment: ase.Atoms | None = None,
    **kwargs,
) -> tuple[tuple[int, ...]]:
    """
    Find all matches of a substructure pattern in a given ASE Atoms object.

    Parameters
    ----------
    atoms : ase.Atoms
        The molecule or structure in which to search for substructure matches.
    smiles : str, optional
        A SMILES string representing the substructure pattern to match.
    smarts : str, optional
        A SMARTS string representing the substructure pattern to match.
    mol : Chem.Mol, optional
        An RDKit Mol object representing the substructure pattern to match.
    fragment : ase.Atoms, optional
        An ASE Atoms object representing the substructure pattern to match.
        If provided, it will be converted to an RDKit Mol object for matching.
    **kwargs
        Additional keyword arguments passed to `ase2rdkit`.

    Returns
    -------
    tuple of tuple of int
        A tuple of atom index tuples, each corresponding to one match of the pattern.
    """
    pattern = None
    if smiles is not None:
        pattern = Chem.MolFromSmiles(smiles)
        pattern = Chem.AddHs(pattern)  # Ensure hydrogens are added for matching
    if smarts is not None:
        if pattern is not None:
            raise ValueError("Can only specify one pattern")
        pattern = Chem.MolFromSmarts(smarts)
    if mol is not None:
        if pattern is not None:
            raise ValueError("Can only specify one pattern")
        pattern = mol
    if fragment is not None:
        if pattern is not None:
            raise ValueError("Can only specify one pattern")
        pattern = ase2rdkit(fragment, **kwargs)
    if pattern is None:
        raise ValueError("Must specify a pattern")

    Chem.SanitizeMol(pattern)

    mol = ase2rdkit(atoms, **kwargs)
    matches = mol.GetSubstructMatches(pattern)
    return matches


def get_substructures(
    atoms: ase.Atoms,
    **kwargs,
) -> list[ase.Atoms]:
    """
    Extract all matched substructures from an ASE Atoms object.

    Parameters
    ----------
    atoms : ase.Atoms
        The structure to search in.
    smarts : str, optional
        A SMARTS string to match substructures.
    smiles : str, optional
        A SMILES string to match substructures.
    mol : Chem.Mol, optional
        An RDKit Mol object to match substructures.
    fragment : ase.Atoms, optional
        A specific ASE Atoms object to match against the structure.
    **kwargs
        Additional keyword arguments passed to `match_substructure`.

    Returns
    -------
    list of ase.Atoms
        List of substructure fragments matching the pattern.
    """
    return [atoms[match] for match in match_substructure(atoms, **kwargs)]


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
