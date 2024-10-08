# from rdkit2ase import pack
import numpy as np
import pytest

from rdkit2ase import pack, smiles2conformers


def test_pack_pbc():
    water = smiles2conformers("O", 1)
    mol_dist = water[0].get_all_distances()
    min_mol_dist = mol_dist[mol_dist > 0].min()

    # pack a box
    atoms = pack([water], [10], 997, 42, tolerance=1.5)

    atoms_dist = atoms.get_all_distances(mic=True)
    assert len(atoms) == np.sum(atoms_dist < min_mol_dist * 0.99)


def test_pack_seeded():
    water = smiles2conformers("O", 1)

    atoms1 = pack([water], [1], 1000, seed=42)
    atoms2 = pack([water], [1], 1000, seed=42)

    assert np.all(atoms1.get_positions() == atoms2.get_positions())

    atoms3 = pack([water], [1], 1000, seed=43)
    assert not np.all(atoms1.get_positions() == atoms3.get_positions())


def test_pack_density():
    ethanol = smiles2conformers("CCO", 1)
    water = smiles2conformers("O", 2)
    hydrochloric_acid = smiles2conformers("Cl", 3)

    atoms = pack([ethanol], [1], density=1000)
    assert atoms.get_chemical_formula() == "C2H6O"

    atoms = pack([ethanol], [2], density=1000)
    assert atoms.get_chemical_formula() == "C4H12O2"

    atoms = pack([ethanol, water, hydrochloric_acid], [2, 1, 3], density=1000)
    assert atoms.get_chemical_formula() == "C4H17Cl3O3"


def test_pack_atoms():
    methane = smiles2conformers("C", 10)

    atoms = pack([methane], [2], density=800)
    assert atoms.get_chemical_formula() == "C2H8"
    assert atoms.get_volume() == pytest.approx(66.6, abs=0.001)
