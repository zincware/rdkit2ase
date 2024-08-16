import io

import ase.io
import rdkit.Chem.AllChem
import rdkit.Chem.rdDetermineBonds
import rdkit.Geometry


def rdkit2ase(mol) -> ase.Atoms:
    """Convert an RDKit molecule to an ASE atoms object."""
    mol = rdkit.Chem.AddHs(mol)
    rdkit.Chem.AllChem.EmbedMolecule(mol)
    rdkit.Chem.AllChem.UFFOptimizeMolecule(mol)

    return ase.Atoms(
        positions=mol.GetConformer().GetPositions(),
        numbers=[atom.GetAtomicNum() for atom in mol.GetAtoms()],
    )


def ase2rdkit(atoms: ase.Atoms) -> rdkit.Chem.Mol:
    """Convert an ASE Atoms object to an RDKit molecule."""
    with io.StringIO() as f:
        ase.io.write(f, atoms, format="xyz")
        f.seek(0)
        xyz = f.read()
        raw_mol = rdkit.Chem.MolFromXYZBlock(xyz)

    mol = rdkit.Chem.Mol(raw_mol)
    rdkit.Chem.rdDetermineBonds.DetermineBonds(
        mol, charge=int(sum(atoms.get_initial_charges()))
    )
    return mol
