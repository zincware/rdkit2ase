import sys

import numpy as np
import pytest
import rdkit.Chem
from ase import Atoms

import rdkit2ase
from rdkit2ase.utils import (
    find_connected_components,
    rdkit_determine_bonds,
    unwrap_molecule,
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
    ["O", "CC", "C1=CC=CC=C1", "C1CCCCC1", "C1=CC=CC=C1O", "F[B-](F)(F)F", "[Li+]", "[Cl-]"],
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
