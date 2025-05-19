# from rdkit2ase import pack
import numpy as np
import numpy.testing as npt
import pytest

from rdkit2ase import ase2rdkit, pack, smiles2conformers


@pytest.mark.parametrize("packmol", ["packmol", "packmol.jl"])
def test_pack_pbc(packmol):
    water = smiles2conformers("O", 1)
    mol_dist = water[0].get_all_distances()
    min_mol_dist = mol_dist[mol_dist > 0].min()

    # pack a box
    atoms = pack([water], [10], 997, 42, tolerance=1.5, packmol=packmol)

    atoms_dist = atoms.get_all_distances(mic=True)
    assert len(atoms) == np.sum(atoms_dist < min_mol_dist * 0.99)


@pytest.mark.parametrize("packmol", ["packmol", "packmol.jl"])
def test_pack_seeded(packmol):
    water = smiles2conformers("O", 1)

    atoms1 = pack([water], [1], 1000, seed=42, packmol=packmol)
    atoms2 = pack([water], [1], 1000, seed=42, packmol=packmol)

    assert np.all(atoms1.get_positions() == atoms2.get_positions())

    atoms3 = pack([water], [1], 1000, seed=43, packmol=packmol)
    assert not np.all(atoms1.get_positions() == atoms3.get_positions())


@pytest.mark.parametrize("packmol", ["packmol", "packmol.jl"])
def test_pack_density(packmol):
    ethanol = smiles2conformers("CCO", 1)
    water = smiles2conformers("O", 2)
    hydrochloric_acid = smiles2conformers("Cl", 3)

    atoms = pack([ethanol], [1], density=1000, packmol=packmol)
    assert atoms.get_chemical_formula() == "C2H6O"

    atoms = pack([ethanol], [2], density=1000, packmol=packmol)
    assert atoms.get_chemical_formula() == "C4H12O2"

    atoms = pack(
        [ethanol, water, hydrochloric_acid], [2, 1, 3], density=1000, packmol=packmol
    )
    assert atoms.get_chemical_formula() == "C4H17Cl3O3"


@pytest.mark.parametrize("packmol", ["packmol", "packmol.jl"])
def test_pack_atoms(packmol):
    methane = smiles2conformers("C", 10)

    atoms = pack([methane], [2], density=800, packmol=packmol)
    assert atoms.get_chemical_formula() == "C2H8"
    assert atoms.get_volume() == pytest.approx(66.6, abs=0.001)


@pytest.mark.parametrize("packmol", ["packmol", "packmol.jl"])
def test_pack_connectivity(packmol):
    water = smiles2conformers("O", 1)
    formaldehyde = smiles2conformers("C=O", 1)
    hydrochloric_acid = smiles2conformers("Cl", 1)

    atoms = pack(
        [formaldehyde, water, hydrochloric_acid],
        [1, 1, 1],
        density=1000,
        packmol=packmol,
    )
    assert atoms.get_chemical_formula() == "CH5ClO2"
    assert atoms.info["connectivity"] == [(0, 1, 2.0), (0, 2, 1.0), (0, 3, 1.0)] + [
        (4, 5, 1.0),
        (4, 6, 1.0),
    ] + [(7, 8, 1.0)]

    npt.assert_array_equal(atoms.get_atomic_numbers(), [6, 8, 1, 1, 8, 1, 1, 17, 1])


@pytest.mark.parametrize("packmol", ["packmol", "packmol.jl"])
def test_pack_charges(packmol):
    water = smiles2conformers("O", 1)
    sodiumcyanide = smiles2conformers("[C-]#N.[Na+]", 1)
    glycine = smiles2conformers("[NH3+]CC([O-])=O", 1)  # small zwitterion

    atoms = pack(
        [water, sodiumcyanide, glycine],
        [1, 1, 1],
        density=1000,
        packmol=packmol,
    )
    npt.assert_array_equal(
        atoms.get_atomic_numbers(), [8, 1, 1, 6, 7, 11] + [7, 6, 6, 8, 8, 1, 1, 1, 1, 1]
    )
    npt.assert_array_equal(
        atoms.get_initial_charges(),
        [0, 0, 0, -1, 0, 1] + [1, 0, 0, -1, 0, 0, 0, 0, 0, 0],
    )

    # go back to molecule

    mol = ase2rdkit(atoms)
    assert mol.GetNumAtoms() == len(atoms)
    # check charges
    charges = [atom.GetFormalCharge() for atom in mol.GetAtoms()]
    assert charges == [0, 0, 0, -1, 0, 1] + [1, 0, 0, -1, 0, 0, 0, 0, 0, 0]
