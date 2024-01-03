import rdkit.Chem.AllChem
import rdkit.Geometry
import ase
from rdkit2ase.xyz2mol import xyz2mol


def rdkit2ase(mol) -> ase.Atoms:
    """Convert an RDKit molecule to an ASE atoms object."""
    mol = rdkit.Chem.AddHs(mol)
    rdkit.Chem.AllChem.EmbedMolecule(mol)
    rdkit.Chem.AllChem.UFFOptimizeMolecule(mol)

    return ase.Atoms(
        positions=mol.GetConformer().GetPositions(),
        numbers=[atom.GetAtomicNum() for atom in mol.GetAtoms()],
    )


def ase2rdkit(atoms: ase.Atoms) -> list[rdkit.Chem.Mol]:
    """Convert an ASE Atoms object to an RDKit molecule."""
    mol = xyz2mol(
        atoms=atoms.get_atomic_numbers().tolist(),
        coordinates=atoms.get_positions().tolist(),
    )
    return mol[0]
