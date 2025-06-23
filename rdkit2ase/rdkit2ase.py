import io
import warnings
from functools import reduce

import ase.io
import numpy as np
import rdkit.Chem.AllChem
import rdkit.Chem.rdDetermineBonds
from ase.build import separate as ase_separate
from rdkit import Chem

from rdkit2ase.connectivity import bond_type_from_order, reconstruct_bonds_from_template
from rdkit2ase.utils import unwrap_molecule


def rdkit2ase(mol, seed: int = 42) -> ase.Atoms:
    """Convert an RDKit molecule to an ASE atoms object."""
    smiles = Chem.MolToSmiles(mol)
    mol = rdkit.Chem.AddHs(mol)
    charges = [atom.GetFormalCharge() for atom in mol.GetAtoms()]
    rdkit.Chem.AllChem.EmbedMolecule(mol, randomSeed=seed)

    atoms = ase.Atoms(
        positions=mol.GetConformer().GetPositions(),
        numbers=[atom.GetAtomicNum() for atom in mol.GetAtoms()],
    )
    atoms.info["smiles"] = smiles
    atoms.info["connectivity"] = []
    for bond in mol.GetBonds():
        a1 = bond.GetBeginAtomIdx()
        a2 = bond.GetEndAtomIdx()
        order = bond.GetBondTypeAsDouble()
        atoms.info["connectivity"].append((a1, a2, order))
    # check if any of the charges are not zero
    if any(charge != 0 for charge in charges):
        atoms.set_initial_charges(charges)
    return atoms


def ase2rdkit(
    atoms: ase.Atoms, separate: bool = True, suggestions: list[str] | None = None
) -> rdkit.Chem.Mol:
    """Convert an ASE Atoms object to an RDKit molecule.

    Parameters
    ----------
    atoms : ase.Atoms
        The ASE Atoms object to convert.
    separate : bool, optional
        separate the "atoms" object using ase.build.separate()
        for faster conversion. Very useful for large systems,
        not necessary for small systems.
    suggestions : list[str] | None, optional
        A list of SMILES strings to use as suggestions for bond determination.
        If provided, RDKit will try to match the connectivity based on these SMILES.
    """
    suggestions = suggestions or []
    if "connectivity" in atoms.info:
        if len(suggestions) > 0:
            warnings.warn(
                f"{suggestions = } will be ignored because "
                "connectivity information is already present in the atoms object.",
                UserWarning,
            )

        mol = Chem.RWMol()
        atom_idx_map = []
        charges = atoms.get_initial_charges()

        for z, charge in zip(atoms.get_chemical_symbols(), charges):
            atom = Chem.Atom(z)
            atom.SetFormalCharge(int(charge))
            idx = mol.AddAtom(atom)
            atom_idx_map.append(idx)

        if isinstance(atoms.info["connectivity"], np.ndarray):
            con = atoms.info["connectivity"].tolist()
        else:
            con = atoms.info["connectivity"]
        for a, b, bond_order in con:
            bond_type = bond_type_from_order(bond_order)
            mol.AddBond(int(a), int(b), bond_type)

        mol = mol.GetMol()
    elif separate:
        mols = []
        charge = 0

        for sub_structure in ase_separate(atoms):
            sub_structure: ase.Atoms
            # center the sub-structure, remove pbc
            unwrapped_sub_structure = unwrap_molecule(sub_structure)
            success = False
            for suggestion in suggestions:
                # Try to reconstruct the bonds from the template
                try:
                    connectivity, charges = reconstruct_bonds_from_template(
                        unwrapped_sub_structure, suggestion
                    )
                    success = True

                    mol = Chem.RWMol()
                    atom_idx_map = []
                    for z, charge in zip(
                        unwrapped_sub_structure.get_chemical_symbols(),
                        charges,
                    ):
                        atom = Chem.Atom(z)
                        atom.SetFormalCharge(int(charge))
                        idx = mol.AddAtom(atom)
                        atom_idx_map.append(idx)

                    for a, b, bond_order in connectivity:
                        bond_type = bond_type_from_order(bond_order)
                        mol.AddBond(int(a), int(b), bond_type)

                    mol = mol.GetMol()
                    mols.append(mol)

                    break
                except ValueError:
                    continue
            if success:
                continue

            with io.StringIO() as f:
                ase.io.write(f, unwrapped_sub_structure, format="xyz")
                f.seek(0)
                xyz = f.read()
                raw_mol = rdkit.Chem.MolFromXYZBlock(xyz)
            mol = rdkit.Chem.Mol(raw_mol)
            for charge in [0, 1, -1, 2, -2]:
                try:
                    rdkit.Chem.rdDetermineBonds.DetermineBonds(
                        mol,
                        charge=int(sum(sub_structure.get_initial_charges())) + charge,
                    )
                    # TODO: set positions of the sub-structure (not unwrapped)
                    mols.append(mol)
                    break
                except ValueError:
                    pass
            else:
                raise ValueError(
                    "Failed to determine bonds for sub-structure up to charge "
                    f"{sum(sub_structure.get_initial_charges()) + charge}"
                    f"and {sub_structure.get_chemical_symbols()}"
                )
        mol = reduce(Chem.CombineMols, mols)
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
