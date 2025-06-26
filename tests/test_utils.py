import sys

import numpy as np
import pytest
import rdkit.Chem
from ase import Atoms

import rdkit2ase
from rdkit2ase.utils import (
    find_connected_components,
    rdkit_determine_bonds,
    suggestions2networkx,
    unwrap_molecule,
)


class SMILES:
    PF6: str = "F[P-](F)(F)(F)(F)F"
    Li: str = "[Li+]"
    EC: str = "C1COC(=O)O1"
    EMC: str = "CCOC(=O)OC"


@pytest.fixture
def ec_emc_li_pf6():
    atoms_pf6 = rdkit2ase.smiles2conformers(SMILES.PF6, numConfs=10)
    atoms_li = rdkit2ase.smiles2conformers(SMILES.Li, numConfs=10)
    atoms_ec = rdkit2ase.smiles2conformers(SMILES.EC, numConfs=10)
    atoms_emc = rdkit2ase.smiles2conformers(SMILES.EMC, numConfs=10)

    return rdkit2ase.pack(
        data=[atoms_pf6, atoms_li, atoms_ec, atoms_emc],
        counts=[3, 3, 8, 12],
        density=1400,
        packmol="packmol.jl",
    )


def make_molecule(shift=(0, 0, 0), cell=(10, 10, 10), pbc=True):
    """Simple diatomic molecule with optional shift and PBC wrapping."""
    pos = np.array([[0.0, 0.0, 0.0], [1.1, 0.0, 0.0]])  # bond length 1.1
    pos += shift
    atoms = Atoms("H2", positions=pos, cell=cell, pbc=pbc)
    return atoms


@pytest.mark.parametrize(
    "shift,expected",
    [
        ((0, 0, 0), 1.1),  # Normal
        ((9.5, 0, 0), 1.1),  # Cross x boundary
        ((0, 9.5, 0), 1.1),  # Cross y boundary
        ((0, 0, 9.5), 1.1),  # Cross z boundary
        ((9.5, 9.5, 9.5), 1.1),  # Cross all boundaries
    ],
)
def test_unwrap_diatomic(shift, expected):
    """Ensure diatomics get unwrapped correctly across boundaries."""
    atoms = make_molecule(shift=shift)
    atoms_wrapped = atoms.copy()
    atoms_unwrapped = unwrap_molecule(atoms_wrapped)

    dist = atoms_unwrapped.get_distance(0, 1, mic=True)
    assert np.isclose(dist, expected, atol=0.05)


def test_multiple_molecules():
    """Test two molecules, one crossing a boundary, one inside."""
    mol1 = make_molecule(shift=(9.5, 0, 0))  # Wrapped
    mol2 = make_molecule(shift=(5, 5, 5))  # Inside
    atoms = mol1 + mol2
    atoms.set_cell([10, 10, 10])
    atoms.set_pbc([True, True, True])

    atoms_unwrapped = unwrap_molecule(atoms.copy())

    # Check both H-H distances are ~1.1
    d1 = atoms_unwrapped.get_distance(0, 1, mic=True)
    d2 = atoms_unwrapped.get_distance(2, 3, mic=True)

    assert np.isclose(d1, 1.1, atol=0.05)
    assert np.isclose(d2, 1.1, atol=0.05)


def test_wrapping_corner_case():
    """Molecule across 3D corner (x, y, z all wrapped)."""
    # Place one atom near box corner, the other offset by 1.1 Å along the diagonal
    a1 = np.array([9.9, 9.9, 9.9])
    bond_length = 1.1 / np.sqrt(3)
    a2 = (a1 + np.array([bond_length] * 3)) % 10.0  # wrapped across corner

    atoms = Atoms("H2", positions=[a1, a2], cell=[10, 10, 10], pbc=True)

    atoms_unwrapped = unwrap_molecule(atoms.copy())
    dist = atoms_unwrapped.get_distance(0, 1, mic=True)

    assert np.isclose(dist, 1.1, atol=0.05)


def test_no_pbc():
    """Ensure no changes when system has no PBC."""
    atoms = make_molecule()
    atoms.set_pbc(False)

    atoms_unwrapped = unwrap_molecule(atoms.copy())
    dist = atoms_unwrapped.get_distance(0, 1)

    assert np.isclose(dist, 1.1, atol=0.05)


def test_idempotent():
    """Calling unwrap on an already unwrapped structure should do nothing."""
    atoms = make_molecule(shift=(0, 0, 0))
    unwrapped = unwrap_molecule(atoms.copy())
    unwrapped2 = unwrap_molecule(unwrapped.copy())

    assert np.allclose(unwrapped.get_positions(), unwrapped2.get_positions())


@pytest.mark.parametrize("networkx", [True, False])
def test_find_connected_components_networkx(monkeypatch, networkx):
    if not networkx:
        monkeypatch.setitem(sys.modules, "networkx", None)

    connectivity = [
        (0, 1, 1.0),
        (0, 1, 1.0),  # duplicate edge
        (1, 2, 1.0),
        (4, 3, 1.0),
        (4, 5, 1.0),
        (6, 7, 1.0),
    ]

    components = list(find_connected_components(connectivity))
    assert len(components) == 3
    assert set(components[0]) == {0, 1, 2}
    assert set(components[1]) == {3, 4, 5}
    assert set(components[2]) == {6, 7}


@pytest.mark.parametrize(
    "smiles",
    [
        # Simple neutral molecules
        "O",  # Water
        "CC",  # Ethane
        "C1CCCCC1",  # Cyclohexane
        "C1=CC=CC=C1",  # Benzene
        "C1=CC=CC=C1O",  # Phenol
        # Simple anions/cations
        "[Li+]",  # Lithium ion
        "[Na+]",  # Sodium ion
        "[Cl-]",  # Chloride
        "[OH-]",  # Hydroxide
        "[NH4+]",  # Ammonium
        "[CH3-]",  # Methyl anion
        "[C-]#N",  # Cyanide anion"
        # Phosphate and sulfate groups
        "OP(=O)(O)O ",  # H3PO4
        "OP(=O)(O)[O-]",  # H2PO4-
        "[O-]P(=O)(O)[O-]",  # HPO4 2-
        "[O-]P(=O)([O-])[O-]",  # PO4 3-
        "OP(=O)=O",  # HPO3
        "[O-]P(=O)=O",  # PO3 -
        "OS(=O)(=O)O",  # H2SO4
        "OS(=O)(=O)[O-]",  # HSO4-
        "[O-]S(=O)(=O)[O-]",  # SO4 2-
        "[O-]S(=O)(=O)([O-])",  # SO3 2-
        # Multiply charged ions
        # "[Fe+3]",            # Iron(III)
        # "[Fe++]",            # Iron(II) alternative syntax
        # "[O-2]",             # Oxide dianion
        # "[Mg+2]",            # Magnesium ion
        # "[Ca+2]",            # Calcium ion
        # Charged organic fragments
        "C[N+](C)(C)C",  # Tetramethylammonium
        # "[N-]=[N+]=[N-]",       # Azide ion
        "C1=[N+](C=CC=C1)[O-]",  # Nitrobenzene
        # Complex anions
        "F[B-](F)(F)F",  # Tetrafluoroborate
        "F[P-](F)(F)(F)(F)F",  # Hexafluorophosphate
        "[O-]C(=O)C(=O)[O-]",  # Oxalate dianion
        # Zwitterions
        "C(C(=O)[O-])N",  # Glycine
        "C1=CC(=CC=C1)[N+](=O)[O-]",  # Nitrobenzene
        # Aromatic heterocycles
        "c1ccncc1",  # Pyridine
        "c1cccnc1",  # Pyrimidine
        # Polyaromatics
        "c1ccc2ccccc2c1",  # Naphthalene
        "c1ccc2c(c1)ccc3c2cccc3",  # Phenanthrene
    ],
)
def test_rdkit_determine_bonds(smiles: str):
    atoms = rdkit2ase.smiles2atoms(smiles)
    del atoms.info["connectivity"]
    del atoms.info["smiles"]
    mol = rdkit_determine_bonds(atoms)

    target_mol = rdkit.Chem.MolFromSmiles(smiles)
    target_mol = rdkit.Chem.AddHs(target_mol)
    target_smiles = rdkit.Chem.MolToSmiles(target_mol)
    found_smiles = rdkit.Chem.MolToSmiles(mol)

    assert target_smiles == found_smiles


def test_suggestions2networkx():
    graphs = suggestions2networkx(smiles=["C", "C=O"])
    assert len(graphs) == 2
    assert graphs[0].number_of_nodes() == 5
    assert graphs[0].number_of_edges() == 4

    assert graphs[1].number_of_nodes() == 4  # pbc and cell
    assert graphs[1].number_of_edges() == 3
