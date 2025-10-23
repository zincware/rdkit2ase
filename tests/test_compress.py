import pytest

import molify
from molify.utils import calculate_density


@pytest.mark.parametrize("packmol", ["packmol.jl", "packmol"])
def test_compress(packmol):
    water = molify.smiles2conformers("O", 1)
    # pack a box
    atoms = molify.pack([water], [10], density=500, packmol=packmol)

    density = calculate_density(atoms)
    assert density == pytest.approx(500, abs=0.01)

    compressed_atoms = molify.compress(
        atoms,
        density=1000,
    )

    density = calculate_density(compressed_atoms)
    assert density == pytest.approx(1000, abs=0.01)


@pytest.mark.parametrize("packmol", ["packmol.jl", "packmol"])
def test_compress_freeze(packmol):
    water = molify.smiles2conformers("O", 1)
    # pack a box
    atoms = molify.pack([water], [10], density=500, packmol=packmol)

    density = calculate_density(atoms)
    assert density == pytest.approx(500, abs=0.01)

    compressed_atoms = molify.compress(
        atoms,
        density=1000,
        freeze_molecules=True,
    )
    # compressed_atoms = atoms

    density = calculate_density(compressed_atoms)
    assert density == pytest.approx(1000, abs=0.01)
    # iterate all the OH bonds and ensure they are not compressed
    all_molecules = list(molify.iter_fragments(compressed_atoms))
    assert len(all_molecules) == 10
    ref_water = water[0].copy()
    for molecule in all_molecules:
        # get the indices of the O and H atoms
        assert len(molecule) == 3
        for idx, atom in enumerate(molecule):
            if atom.symbol == "O":
                # test the O-H distance
                if idx == 0:
                    assert molecule.get_distance(0, 1) == pytest.approx(
                        ref_water.get_distance(0, 1), abs=0.1
                    )
                else:
                    assert molecule.get_distance(1, 2) == pytest.approx(
                        ref_water.get_distance(0, 1), abs=0.1
                    )
