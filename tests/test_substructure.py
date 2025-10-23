import pytest
from rdkit import Chem

import molify


def test_match_substructure():
    atoms = molify.smiles2atoms("CC(=O)O")

    # match CH3 fragment using smarts
    match = molify.match_substructure(atoms, smarts="[C]([H])([H])[H]")
    assert match == ((0, 4, 5, 6),)
    assert atoms[match[0]].get_chemical_symbols() == ["C", "H", "H", "H"]

    # now match using a ase.Atoms object

    ref = molify.smiles2atoms("[C]([H])([H])[H]")
    match = molify.match_substructure(atoms, fragment=ref)
    assert match == ((0, 4, 5, 6),)

    # now match using a Chem.Mol object
    ref_mol = Chem.MolFromSmarts("[C]([H])([H])[H]")
    match = molify.match_substructure(atoms, mol=ref_mol)
    assert match == ((0, 4, 5, 6),)

    # check everything else raises TypeError
    with pytest.raises(TypeError):
        molify.match_substructure(atoms, 42)  # type: ignore[arg-type]


@pytest.mark.parametrize("packmol", ["packmol.jl"])
def test_match_substructur_box(packmol):
    atoms = molify.smiles2conformers("CC(=O)O", 1)
    box = molify.pack([atoms], counts=[3], packmol=packmol, density=0.5)

    # match CH3 fragment using smarts
    match = molify.match_substructure(box, smarts="[C]([H])([H])[H]")
    assert match == ((0, 4, 5, 6), (8, 12, 13, 14), (16, 20, 21, 22))
    for m in match:
        assert box[m].get_chemical_symbols() == ["C", "H", "H", "H"]

    # now match using a ase.Atoms object
    ref = molify.smiles2atoms("[C]([H])([H])[H]")
    match = molify.match_substructure(box, fragment=ref)
    assert match == ((0, 4, 5, 6), (8, 12, 13, 14), (16, 20, 21, 22))

    # now match using a Chem.Mol object
    ref_mol = Chem.MolFromSmarts("[C]([H])([H])[H]")
    match = molify.match_substructure(box, mol=ref_mol)
    assert match == ((0, 4, 5, 6), (8, 12, 13, 14), (16, 20, 21, 22))

    # check everything else raises TypeError
    with pytest.raises(TypeError):
        molify.match_substructure(box, 42)


def test_get_substructure():
    atoms = molify.smiles2atoms("C(C(CO[N+](=O)[O-])O[N+](=O)[O-])O[N+](=O)[O-]")
    # match NO3 group using smarts
    frames = molify.get_substructures(atoms, smarts="[N+](=O)[O-]")
    assert len(frames) == 3
    for frame in frames:
        assert frame.get_chemical_symbols() == ["N", "O", "O"]

    # match using a ase.Atoms object
    ref = molify.smiles2atoms("[N+](=O)[O-]")
    frames = molify.get_substructures(atoms, fragment=ref)
    assert len(frames) == 3
    for frame in frames:
        assert frame.get_chemical_symbols() == ["N", "O", "O"]

    # match using a Chem.Mol object
    ref_mol = Chem.MolFromSmarts("[N+](=O)[O-]")
    frames = molify.get_substructures(atoms, mol=ref_mol)
    assert len(frames) == 3
    for frame in frames:
        assert frame.get_chemical_symbols() == ["N", "O", "O"]


@pytest.mark.parametrize("packmol", ["packmol.jl"])
def test_get_substructure_box(packmol):
    atoms = molify.smiles2conformers(
        "C(C(CO[N+](=O)[O-])O[N+](=O)[O-])O[N+](=O)[O-]", 1
    )
    box = molify.pack([atoms], counts=[3], packmol=packmol, density=0.5)

    # match NO3 group using smarts
    frames = molify.get_substructures(box, smarts="[N+](=O)[O-]")
    assert len(frames) == 9
    for frame in frames:
        assert frame.get_chemical_symbols() == ["N", "O", "O"]

    # match using a ase.Atoms object
    ref = molify.smiles2atoms("[N+](=O)[O-]")
    frames = molify.get_substructures(box, fragment=ref)
    assert len(frames) == 9
    for frame in frames:
        assert frame.get_chemical_symbols() == ["N", "O", "O"]

    # match using a Chem.Mol object
    ref_mol = Chem.MolFromSmarts("[N+](=O)[O-]")
    frames = molify.get_substructures(box, mol=ref_mol)
    assert len(frames) == 9
    for frame in frames:
        assert frame.get_chemical_symbols() == ["N", "O", "O"]


@pytest.mark.parametrize("remove_connectivity", [True, False])
@pytest.mark.parametrize("packmol", ["packmol.jl"])
def test_iter_fragments(packmol, remove_connectivity):
    water = molify.smiles2conformers("O", 1)
    box = molify.pack([water], [10], density=500, packmol=packmol)
    if remove_connectivity:
        del box.info["connectivity"]
    fragments = list(molify.iter_fragments(box))
    assert len(fragments) == 10
    for atoms in fragments:
        assert len(atoms) == 3


def test_bmim_bf4_no_info():
    bmim = molify.smiles2conformers("CCCCN1C=C[N+](=C1)C", numConfs=1)
    bf4 = molify.smiles2conformers("[B-](F)(F)(F)F", numConfs=1)

    box = molify.pack(
        [bmim, bf4],
        counts=[10, 10],
        density=500,
        packmol="packmol.jl",
        tolerance=3,
    )
    del box.info["connectivity"]
    bf4_matches = molify.match_substructure(
        box, smiles="[B-](F)(F)(F)F", suggestions=[]
    )
    assert len(bf4_matches) == 10
    for match in bf4_matches:
        assert box[match].get_chemical_symbols() == bf4[0].get_chemical_symbols()

    bmim_matches = molify.match_substructure(
        box,
        smiles="CCCCN1C=C[N+](=C1)C",
        suggestions=[],
    )
    assert len(bmim_matches) == 10
    for match in bmim_matches:
        assert len(match) == 25
        assert sorted(box[match].get_chemical_symbols()) == sorted(
            bmim[0].get_chemical_symbols()
        )

    bmim_matches = molify.match_substructure(
        box,
        smarts="[H]c1c([H])[n+](C([H])([H])[H])c([H])n1C([H])([H])C([H])([H])C([H])([H])C([H])([H])[H]",
        suggestions=[],
    )
    assert len(bmim_matches) == 10
    for match in bmim_matches:
        assert len(match) == 25
        assert sorted(box[match].get_chemical_symbols()) == sorted(
            bmim[0].get_chemical_symbols()
        )


@pytest.fixture
def ethanol_mol():
    """Returns an RDKit molecule for ethanol (CCO) with explicit hydrogens."""
    mol = Chem.MolFromSmiles("CCO")
    return Chem.AddHs(mol)


@pytest.fixture
def toluene_mol():
    """Returns an RDKit molecule for toluene (Cc1ccccc1) with explicit hydrogens."""
    mol = Chem.MolFromSmiles("Cc1ccccc1")
    return Chem.AddHs(mol)


# =============================================================================
# Test Cases for select_atoms_flat_unique
# =============================================================================


def test_select_carbons(ethanol_mol):
    """Test selecting all carbon atoms."""
    # Ethanol (CCO with Hs): C(0), C(1), O(2), H(3-8)
    indices = molify.select_atoms_flat_unique(ethanol_mol, "[#6]")
    assert sorted(indices) == [0, 1]


def test_select_oxygen(ethanol_mol):
    """Test selecting the oxygen atom."""
    indices = molify.select_atoms_flat_unique(ethanol_mol, "[#8]")
    assert sorted(indices) == [2]


def test_no_matches(ethanol_mol):
    """Test a SMARTS pattern that has no matches."""
    indices = molify.select_atoms_flat_unique(ethanol_mol, "[F]")  # Fluorine
    assert indices == []


# --- Hydrogen Handling Tests ---


def test_hydrogens_excluded_by_default(ethanol_mol):
    """Test that hydrogens are excluded by default."""
    # C-O bond involves atoms 1 and 2. Hydrogens attached are not included.
    indices = molify.select_atoms_flat_unique(ethanol_mol, "CO")
    assert sorted(indices) == [1, 2]


def test_hydrogens_included(ethanol_mol):
    """Test the 'include' option for hydrogens."""
    # C-O bond (atoms 1, 2) plus attached hydrogens (atoms 6, 7, 8)
    # H on O is atom 8. Hs on C(1) are 6, 7.
    indices = molify.select_atoms_flat_unique(ethanol_mol, "CO", hydrogens="include")
    # Expected: C(1), O(2), H(6), H(7), H(8)
    assert sorted(indices) == [1, 2, 6, 7, 8]


def test_hydrogens_isolated(ethanol_mol):
    """Test the 'isolated' option for hydrogens."""
    # Select ONLY the hydrogens from the C-O match
    indices = molify.select_atoms_flat_unique(
        ethanol_mol, "CO", hydrogens="isolated"
    )
    # Expected: H(6), H(7), H(8)
    assert sorted(indices) == [6, 7, 8]


def test_smarts_with_explicit_hydrogens(ethanol_mol):
    """Test a SMARTS pattern that explicitly includes hydrogens."""
    # Find all hydrogens attached to an oxygen
    indices = molify.select_atoms_flat_unique(
        ethanol_mol, "[#8]-[H]", hydrogens="include"
    )
    # Expected: O(2), H(8)
    assert sorted(indices) == [2, 8]

    # Now isolate only the hydrogen from that match
    h_indices = molify.select_atoms_flat_unique(
        ethanol_mol, "[#8]-[H]", hydrogens="isolated"
    )
    assert sorted(h_indices) == [8]


# --- Mapped SMILES Tests ---


def test_mapped_smiles(ethanol_mol):
    """Test selecting only mapped atoms using a mapped SMILES pattern."""
    # The pattern "[C:1][C:2]O" matches atoms 0, 1, and 2,
    # but only C:1 and C:2 are mapped.
    # The function should now return only the indices of the mapped atoms.
    indices = molify.select_atoms_flat_unique(ethanol_mol, "[C:1][C:2]O")
    # FIX: The test's expectation is updated to only expect the mapped carbons [0, 1].
    assert sorted(indices) == [0, 1]


def test_mapped_smiles_with_hydrogens(ethanol_mol):
    """Test mapped SMILES with hydrogen filtering."""
    # Pattern "C[O:1]" matches atoms C(1) and O(2), but only O(2) is mapped.
    # The core selection will be just atom 2.

    # Include hydrogens attached to the mapped oxygen
    indices_included = molify.select_atoms_flat_unique(
        ethanol_mol, "C[O:1]", hydrogens="include"
    )
    # Expected: O(2) and its hydrogen H(8)
    assert sorted(indices_included) == [2, 8]

    # Exclude hydrogens (returns just the mapped heavy atom)
    indices_excluded = molify.select_atoms_flat_unique(
        ethanol_mol, "C[O:1]", hydrogens="exclude"
    )
    assert sorted(indices_excluded) == [2]

    # Isolate only hydrogens attached to the mapped oxygen
    indices_isolated = molify.select_atoms_flat_unique(
        ethanol_mol, "C[O:1]", hydrogens="isolated"
    )
    assert sorted(indices_isolated) == [8]


# --- Error Handling Tests ---


def test_invalid_smarts_raises_error():
    """Test that an invalid SMARTS string raises a ValueError."""
    mol = Chem.MolFromSmiles("C")
    with pytest.raises(ValueError, match="Invalid SMARTS/SMILES"):
        molify.select_atoms_flat_unique(mol, "this is not valid")


# =============================================================================
# Test Cases for visualize_selected_molecules
# =============================================================================


def test_visualize_selected_molecules_basic(ethanol_mol):
    """Test basic visualization functionality."""
    # Select some atoms to highlight
    a = [0, 1]  # Carbons
    b = [2]  # Oxygen

    img = molify.visualize_selected_molecules(ethanol_mol, a, b)
    assert img is not None


def test_visualize_selected_molecules_empty_selections(ethanol_mol):
    """Test visualization with empty selections - shows molecule without highlights."""
    img = molify.visualize_selected_molecules(ethanol_mol)
    assert img is not None


def test_visualize_selected_molecules_overlapping_selections(ethanol_mol):
    """Test visualization with overlapping selections (later args take precedence)."""
    a = [0, 1, 2]  # All heavy atoms
    b = [2]  # Oxygen (should get color from b, not a)

    img = molify.visualize_selected_molecules(ethanol_mol, a, b)
    assert img is not None


def test_visualize_selected_molecules_single_selection(ethanol_mol):
    """Test visualization with a single selection."""
    a = [0, 1]  # Carbons

    img = molify.visualize_selected_molecules(ethanol_mol, a)
    assert img is not None


def test_visualize_selected_molecules_multiple_selections(ethanol_mol):
    """Test visualization with multiple selections."""
    a = [0]  # First carbon
    b = [1]  # Second carbon
    c = [2]  # Oxygen

    img = molify.visualize_selected_molecules(ethanol_mol, a, b, c)
    assert img is not None


def test_visualize_selected_molecules_with_alpha(ethanol_mol):
    """Test visualization with custom alpha value."""
    a = [0, 1]  # Carbons
    b = [2]  # Oxygen

    # Test with different alpha values
    img_transparent = molify.visualize_selected_molecules(
        ethanol_mol, a, b, alpha=0.2
    )
    assert img_transparent is not None

    img_opaque = molify.visualize_selected_molecules(ethanol_mol, a, b, alpha=1.0)
    assert img_opaque is not None


def test_select_atoms_grouped(ethanol_water):
    """Test selecting atoms from disconnected fragments."""
    # ethanol_water is an ase.Atoms object with 2 ethanol and 2 water molecules.
    # We need to convert it to an RDKit Mol object first.
    mol = molify.ase2rdkit(ethanol_water)

    # Test 1: Select all carbon atoms.
    # Should find 2 groups (the two ethanol molecules), each with 2 carbons.
    indices_carbons = molify.select_atoms_grouped(mol, "[#6]")
    assert indices_carbons == [[0, 1], [9, 10]]
    # Verify they are indeed carbons
    for group in indices_carbons:
        for idx in group:
            assert mol.GetAtomWithIdx(idx).GetAtomicNum() == 6

    # Test 2: Select all oxygen atoms.
    # Should find 4 groups (2 ethanol, 2 water), each with 1 oxygen.
    indices_oxygens = molify.select_atoms_grouped(mol, "[#8]")
    assert indices_oxygens == [[2], [11], [18], [21]]
    # Verify they are indeed oxygens
    for group in indices_oxygens:
        for idx in group:
            assert mol.GetAtomWithIdx(idx).GetAtomicNum() == 8

    # Test 3: Select a non-existent atom.
    indices_fluorine = molify.select_atoms_grouped(mol, "[F]")
    assert indices_fluorine == []

    # Test 4: Select hydroxyl group in ethanol, including hydrogens.
    indices_hydroxyl = molify.select_atoms_grouped(mol, "[OH]", hydrogens="include")
    assert indices_hydroxyl == [[2, 8], [11, 17]]
    for group in indices_hydroxyl:
        symbols = [mol.GetAtomWithIdx(idx).GetSymbol() for idx in group]
        assert "O" in symbols
        assert "H" in symbols


def test_use_label_multiple_times(alanine_dipeptide):
    """Test selecting atoms using the same label multiple times."""
    mol = molify.ase2rdkit(alanine_dipeptide)
    with pytest.raises(ValueError, match="Label '1' is used multiple times"):
        # we do not allow the same label to be used multiple for selection,
        # need to be unique
        molify.select_atoms_grouped(
            mol, smarts_or_smiles="CC(=O)N[C:1]([C:1])C(=O)NC"
        )


def test_select_atoms_grouped_order(alanine_dipeptide):
    mol = molify.ase2rdkit(alanine_dipeptide)
    indices = molify.select_atoms_grouped(
        mol, smarts_or_smiles="CC(=O)N[C:1](C)C(=O)NC"
    )
    assert indices == [[4]]

    indices = molify.select_atoms_grouped(
        mol, smarts_or_smiles="CC(=O)NC([C:1])C(=O)NC"
    )
    assert indices == [[5]]

    indices = molify.select_atoms_grouped(
        mol, smarts_or_smiles="CC(=O)NC(C)[C:1](=O)NC"
    )
    assert indices == [[6]]

    indices = molify.select_atoms_grouped(
        mol, smarts_or_smiles="CC(=O)NC(C)C(=O)[N:1]C"
    )
    assert indices == [[8]]

    # now all of them
    indices = molify.select_atoms_grouped(
        mol, smarts_or_smiles="CC(=O)N[C:1]([C:2])[C:3](=O)[N:4]C"
    )
    assert indices == [[4, 5, 6, 8]]
    # now in a different order
    indices = molify.select_atoms_grouped(
        mol, smarts_or_smiles="CC(=O)N[C:4]([C:3])[C:2](=O)[N:1]C"
    )
    assert indices == [[8, 6, 5, 4]]

    # now with hydrogens which are in order 1, hydrogens of 1, 2 hydrogens of 2, ...
    indices = molify.select_atoms_grouped(
        mol, smarts_or_smiles="CC(=O)N[C:1](C)C(=O)NC", hydrogens="include"
    )
    assert indices == [[4, 14]]

    indices = molify.select_atoms_grouped(
        mol, smarts_or_smiles="CC(=O)N[C:1]([C:2])C(=O)NC", hydrogens="include"
    )
    assert indices == [[4, 14, 5, 15, 16, 17]]

    # now inverse order
    indices = molify.select_atoms_grouped(
        mol, smarts_or_smiles="CC(=O)N[C:2]([C:1])C(=O)NC", hydrogens="include"
    )
    assert indices == [[5, 15, 16, 17, 4, 14]]

    # now with hydrogens isolated
    indices = molify.select_atoms_grouped(
        mol, smarts_or_smiles="CC(=O)N[C:1]([C:2])C(=O)NC", hydrogens="isolated"
    )
    assert indices == [[14, 15, 16, 17]]
    indices = molify.select_atoms_grouped(
        mol, smarts_or_smiles="CC(=O)N[C:2]([C:1])C(=O)NC", hydrogens="isolated"
    )
    assert indices == [[15, 16, 17, 14]]


def test_select_atoms_grouped_order_box(alanine_dipeptide_box):
    mol = molify.ase2rdkit(alanine_dipeptide_box)
    indices = molify.select_atoms_grouped(
        mol, smarts_or_smiles="CC(=O)N[C:1](C)C(=O)NC"
    )
    assert indices == [[4], [26], [48]]

    indices = molify.select_atoms_grouped(
        mol, smarts_or_smiles="CC(=O)NC([C:1])C(=O)NC"
    )
    assert indices == [[5], [27], [49]]

    indices = molify.select_atoms_grouped(
        mol, smarts_or_smiles="CC(=O)NC(C)[C:1](=O)NC"
    )
    assert indices == [[6], [28], [50]]

    indices = molify.select_atoms_grouped(
        mol, smarts_or_smiles="CC(=O)NC(C)C(=O)[N:1]C"
    )
    assert indices == [[8], [30], [52]]

    # now all of them
    indices = molify.select_atoms_grouped(
        mol, smarts_or_smiles="CC(=O)N[C:1]([C:2])[C:3](=O)[N:4]C"
    )
    assert indices == [[4, 5, 6, 8], [26, 27, 28, 30], [48, 49, 50, 52]]
    # now in a different order
    indices = molify.select_atoms_grouped(
        mol, smarts_or_smiles="CC(=O)N[C:4]([C:3])[C:2](=O)[N:1]C"
    )
    assert indices == [[8, 6, 5, 4], [30, 28, 27, 26], [52, 50, 49, 48]]

    # now with hydrogens which are in order 1, hydrogens of 1, 2 hydrogens of 2, ...
    indices = molify.select_atoms_grouped(
        mol, smarts_or_smiles="CC(=O)N[C:1](C)C(=O)NC", hydrogens="include"
    )
    assert indices == [[4, 14], [26, 36], [48, 58]]

    indices = molify.select_atoms_grouped(
        mol, smarts_or_smiles="CC(=O)N[C:1]([C:2])C(=O)NC", hydrogens="include"
    )
    assert indices == [
        [4, 14, 5, 15, 16, 17],
        [26, 36, 27, 37, 38, 39],
        [48, 58, 49, 59, 60, 61],
    ]

    # now inverse order
    indices = molify.select_atoms_grouped(
        mol, smarts_or_smiles="CC(=O)N[C:2]([C:1])C(=O)NC", hydrogens="include"
    )
    assert indices == [
        [5, 15, 16, 17, 4, 14],
        [27, 37, 38, 39, 26, 36],
        [49, 59, 60, 61, 48, 58],
    ]

    # now with hydrogens isolated
    indices = molify.select_atoms_grouped(
        mol, smarts_or_smiles="CC(=O)N[C:1]([C:2])C(=O)NC", hydrogens="isolated"
    )
    assert indices == [[14, 15, 16, 17], [36, 37, 38, 39], [58, 59, 60, 61]]
    indices = molify.select_atoms_grouped(
        mol, smarts_or_smiles="CC(=O)N[C:2]([C:1])C(=O)NC", hydrogens="isolated"
    )
    assert indices == [[15, 16, 17, 14], [37, 38, 39, 36], [59, 60, 61, 58]]


@pytest.mark.parametrize("packmol", ["packmol.jl"])
def test_select_atoms_grouped_ions_with_none_bond_orders(packmol):
    """Test that ase2rdkit automatically handles None bond orders.

    This reproduces the exact error scenario from hillclimber where
    connectivity exists but has None bond orders, causing:
    ValueError: Edge (0, 1) is missing 'bond_order' attribute.

    The fix: ase2rdkit now automatically detects and handles this.
    """
    # Create a system with water and ions (reproduces user's scenario)
    water = molify.smiles2conformers("O", numConfs=1)
    na_ion = molify.smiles2conformers("[Na+]", numConfs=1)
    cl_ion = molify.smiles2conformers("[Cl-]", numConfs=1)

    # Pack them together
    box = molify.pack(
        [water, na_ion, cl_ion],
        counts=[16, 1, 1],
        density=1000,
        packmol=packmol,
        tolerance=2.0,
    )

    # Simulate corrupted connectivity with None bond orders
    # This causes the exact error the user reported
    box.info["connectivity"] = [(i, j, None) for i, j, _ in box.info["connectivity"]]

    # ase2rdkit should NOW automatically handle this (after our fix)
    mol = molify.ase2rdkit(box)

    # Verify select_atoms_grouped works
    na_indices = molify.select_atoms_grouped(mol, "[Na+]")
    assert len(na_indices) == 1

    cl_indices = molify.select_atoms_grouped(mol, "[Cl-]")
    assert len(cl_indices) == 1

    water_indices = molify.select_atoms_grouped(mol, "[OH2]")
    assert len(water_indices) == 16
