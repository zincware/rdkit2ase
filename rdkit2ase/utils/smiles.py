import ase
from rdkit import Chem
from rdkit.Chem import rdDistGeom

from ..rdkit2ase import rdkit2ase


def smiles2atoms(smiles: str) -> ase.Atoms:
    """
    Convert a SMILES string to an ASE Atoms object.

    Args:
        smiles (str): The SMILES string.

    Returns:
        atoms (ase.Atoms): The Atoms object.
    """
    mol = Chem.MolFromSmiles(smiles)
    return rdkit2ase(mol)


def smiles2conformers(
    smiles: str,
    numConfs: int,  # noqa N803
    randomSeed: int = 42,  # noqa N803
    maxAttempts: int = 1000,  # noqa N803
) -> list[ase.Atoms]:
    """Create multiple conformers for a SMILES string.

    Args:
        smiles (str): The SMILES string.
        numConfs (int): The number of conformers to generate.
        randomSeed (int): The random seed.
        maxAttempts (int): The maximum number of attempts.

    Returns:
        images (list[ase.Atoms]): The list of conformers.
    """
    mol = Chem.MolFromSmiles(smiles)
    mol = Chem.AddHs(mol)
    rdDistGeom.EmbedMultipleConfs(
        mol,
        numConfs=numConfs,
        randomSeed=randomSeed,
        maxAttempts=maxAttempts,
    )

    images: list[ase.Atoms] = []

    for conf in mol.GetConformers():
        atoms = ase.Atoms(
            positions=conf.GetPositions(),
            numbers=[atom.GetAtomicNum() for atom in mol.GetAtoms()],
        )
        images.append(atoms)

    return images
