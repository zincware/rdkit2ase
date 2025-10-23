import ase
import numpy as np
import numpy.testing as npt
import pytest

import molify


def test_com_isolated():
    ethanol = molify.smiles2atoms("CCO")
    ethanol.positions += np.array([1.0, 0.0, 0.0])  # Shift to ensure non-zero COM
    com = molify.get_centers_of_mass(ethanol)
    npt.assert_allclose(
        com.get_center_of_mass(), ethanol.get_center_of_mass(), atol=1e-6
    )
    npt.assert_allclose(com.get_positions()[0], ethanol.get_center_of_mass(), atol=1e-6)
    npt.assert_allclose(ethanol.cell, com.get_cell(), atol=1e-6)
    npt.assert_allclose(ethanol.pbc, com.pbc, atol=1e-6)


def test_com_box():
    ethanol = molify.smiles2atoms("CCO")
    ethanol_com = ethanol.get_center_of_mass()

    box = ase.Atoms()
    # add 5 ethanol molecules, shift them each so they don't overlap and then
    # ensure the COM from molify.get_center_of_mass(ethanol)
    # matched the COM of each individual ethanol
    shifts = np.array([[4 * i, 0, 0] for i in range(5)])
    for shift in shifts:
        ethanol_copy = ethanol.copy()
        ethanol_copy.positions += shift
        box += ethanol_copy
    box_com = molify.get_centers_of_mass(box)
    npt.assert_allclose(
        box_com.get_positions(), [ethanol_com + shift for shift in shifts], atol=1e-6
    )
    npt.assert_allclose(box.cell, box_com.get_cell(), atol=1e-6)
    npt.assert_allclose(box.pbc, box_com.pbc, atol=1e-6)


def test_com_explicitly_wrapped_molecule():
    """
    Tests the COM calculation by creating a box with two water molecules,
    one of which is explicitly wrapped across a periodic boundary.
    """
    # 1. Setup a periodic box and a water molecule template
    cell_dim = 7.0
    box = ase.Atoms(cell=np.eye(3) * cell_dim, pbc=True)
    water = molify.smiles2atoms("O")
    water.positions -= water.get_center_of_mass()
    assert np.allclose(water.get_center_of_mass(), [0.0, 0.0, 0.0])
    # 2. Define the target COM positions for two water molecules
    # The second molecule is placed near the x-boundary to force wrapping.
    expected_coms = np.array(
        [
            [cell_dim * 0.5, cell_dim * 0.5, cell_dim * 0.5],
            [cell_dim * 0.9, cell_dim * 0.5, cell_dim * 0.5],
        ]
    )

    # 3. Create the two molecules and add them to the box
    water1 = water.copy()
    water1.positions += expected_coms[0]

    water2 = water.copy()
    water2.positions += expected_coms[1]

    box += water1
    box += water2

    # 4. Wrap the box. This will break water2 across the boundary.
    box.wrap()

    # 5. Calculate COMs using the function under test
    com_atoms = molify.get_centers_of_mass(box)
    calculated_coms = com_atoms.get_positions()

    # 6. Compare results. The order is not guaranteed, so we sort by a stable
    # axis (x-coordinate in this case) before comparing.
    assert len(calculated_coms) == len(expected_coms)

    sort_idx_calc = np.argsort(calculated_coms[:, 0])
    sort_idx_exp = np.argsort(expected_coms[:, 0])

    sorted_calculated = calculated_coms[sort_idx_calc]
    sorted_expected = expected_coms[sort_idx_exp]

    npt.assert_allclose(sorted_calculated, sorted_expected, atol=1e-6)


@pytest.mark.parametrize("shift", [0, 5])
def test_com_with_packed_system(shift):
    """
    Tests the molecule identification part of the COM function on a dense,
    packed system. It verifies that the correct number of molecules and their
    corresponding masses are found.
    """
    water_template = molify.smiles2conformers("O", 1)
    ethanol_template = molify.smiles2conformers("CCO", 1)

    num_water = 7
    num_ethanol = 5
    density = 900  # kg/m^3

    packed_system = molify.pack(
        [water_template, ethanol_template],
        [num_water, num_ethanol],
        density,
        packmol="packmol.jl",
    )
    packed_system.positions += np.array([shift, shift, shift])
    packed_system.wrap()
    packed_system.info.pop("connectivity", None)

    com_atoms = molify.get_centers_of_mass(packed_system)
    calculated_masses = com_atoms.get_masses()

    expected_molecule_count = num_water + num_ethanol
    assert len(com_atoms) == expected_molecule_count, (
        f"Expected {expected_molecule_count} molecules, but found {len(com_atoms)}"
    )

    water_mass = water_template[0].get_masses().sum()
    ethanol_mass = ethanol_template[0].get_masses().sum()

    expected_masses = sorted([water_mass] * num_water + [ethanol_mass] * num_ethanol)

    npt.assert_allclose(sorted(calculated_masses), expected_masses, atol=1e-6)

    assert sum(com_atoms.get_atomic_numbers() == 1) == num_water
    assert sum(com_atoms.get_atomic_numbers() == 2) == num_ethanol


def test_get_centers_of_mass_species():
    a = molify.smiles2conformers("CO", numConfs=10)
    b = molify.smiles2conformers("CCO", numConfs=10)
    c = molify.smiles2conformers("CCCO", numConfs=10)
    d = molify.smiles2conformers("CCCCO", numConfs=10)

    box = molify.pack(
        [a, b, c, d],
        [1, 2, 3, 4],
        density=786,
        packmol="packmol.jl",
    )

    com = molify.get_centers_of_mass(box)

    assert sum(com.get_atomic_numbers() == 1) == 1
    assert sum(com.get_atomic_numbers() == 2) == 2
    assert sum(com.get_atomic_numbers() == 3) == 3
    assert sum(com.get_atomic_numbers() == 4) == 4
    assert set(com.get_atomic_numbers()) == {1, 2, 3, 4}
