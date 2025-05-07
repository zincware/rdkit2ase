import io

import ase.io
import rdkit.Chem.AllChem
import rdkit.Chem.rdDetermineBonds
from rdkit import Chem


# Map float bond orders to RDKit bond types
def bond_type_from_order(order):
    if order == 1.0:
        return Chem.BondType.SINGLE
    elif order == 2.0:
        return Chem.BondType.DOUBLE
    elif order == 3.0:
        return Chem.BondType.TRIPLE
    elif order == 1.5:
        return Chem.BondType.AROMATIC
    else:
        raise ValueError(f"Unsupported bond order: {order}")


def rdkit2ase(mol, seed: int = 42) -> ase.Atoms:
    """Convert an RDKit molecule to an ASE atoms object."""
    mol = rdkit.Chem.AddHs(mol)
    rdkit.Chem.AllChem.EmbedMolecule(mol, randomSeed=seed)
    rdkit.Chem.AllChem.UFFOptimizeMolecule(mol)

    return ase.Atoms(
        positions=mol.GetConformer().GetPositions(),
        numbers=[atom.GetAtomicNum() for atom in mol.GetAtoms()],
    )


def ase2rdkit(atoms: ase.Atoms) -> rdkit.Chem.Mol:
    """Convert an ASE Atoms object to an RDKit molecule."""
    if "connectivity" in atoms.info:
        mol = Chem.RWMol()
        atom_idx_map = []

        # Add atoms
        for z in atoms.get_chemical_symbols():
            atom = Chem.Atom(z)
            idx = mol.AddAtom(atom)
            atom_idx_map.append(idx)

        # Add bonds
        for a, b, bond_order in atoms.info["connectivity"]:
            bond_type = bond_type_from_order(bond_order)
            mol.AddBond(a, b, bond_type)

        mol = mol.GetMol()
    else:
        # If no connectivity information is available, guess the bonds

        with io.StringIO() as f:
            ase.io.write(f, atoms, format="xyz")
            f.seek(0)
            xyz = f.read()
            raw_mol = rdkit.Chem.MolFromXYZBlock(xyz)

        mol = rdkit.Chem.Mol(raw_mol)
        rdkit.Chem.rdDetermineBonds.DetermineBonds(
            mol, charge=int(sum(atoms.get_initial_charges()))
        )

    # Sanitize the molecule (valence checks, aromaticity, etc.)
    Chem.SanitizeMol(mol)
    return mol
