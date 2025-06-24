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
from rdkit2ase.utils import rdkit_determine_bonds, unwrap_molecule


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


def _connectivtiy2rdkit(atoms: ase.Atoms) -> rdkit.Chem.Mol:
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

    return mol.GetMol()


def ase2rdkit(  # noqa: C901
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

    Notes
    -----
    This is split into three attempts:
        1. If the ASE Atoms object has connectivity information this will be used
        2. If suggestions are provided, they will be used to reconstruct the bonds.
        3. For any remaining atoms or if not suggestions are provided, we use rdkit's
           `rdkit.Chem.rdDetermineBonds.DetermineBonds` to guess the bonds
           based on the positions and atomic numbers.
    """

    # Tests
    # check CCCC before CC or C
    # check if you test e.g. NH3 vs SO3 - they have the same graph but would match differently
    # if a rest stays after the graph was decomposed, use rdkit_determine_bonds
    # if only a bond through a structure that has been cut, try the substructures again. E.g. if CCC - Li - CCC and they are only connected through Li, try to match CCC and Li separately
    # if alkalimetal, always unbound and +1

    suggestions = suggestions or []
    if "connectivity" in atoms.info:
        if len(suggestions) > 0:
            warnings.warn(
                f"{suggestions = } will be ignored because "
                "connectivity information is already present in the atoms object.",
                UserWarning,
            )
        mol = _connectivtiy2rdkit(atoms)

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

            mols.append(rdkit_determine_bonds(unwrapped_sub_structure))
        mol = reduce(Chem.CombineMols, mols)
    else:
        # If no connectivity information is available, guess the bonds
        # TODO: should use the same code as above!
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
