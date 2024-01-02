from ..rdkit2ase import rdkit2ase
from rdkit import Chem
import ase


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