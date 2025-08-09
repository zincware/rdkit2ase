import pytest
from rdkit import Chem

import rdkit2ase


def test_match_substructure():
    atoms = rdkit2ase.smiles2atoms("CC(=O)O")

    # match CH3 fragment using smarts
    match = rdkit2ase.match_substructure(atoms, smarts="[C]([H])([H])[H]")
    assert match == ((0, 4, 5, 6),)
    assert atoms[match[0]].get_chemical_symbols() == ["C", "H", "H", "H"]

    # now match using a ase.Atoms object

    ref = rdkit2ase.smiles2atoms("[C]([H])([H])[H]")
    match = rdkit2ase.match_substructure(atoms, fragment=ref)
    assert match == ((0, 4, 5, 6),)

    # now match using a Chem.Mol object
    ref_mol = Chem.MolFromSmarts("[C]([H])([H])[H]")
    match = rdkit2ase.match_substructure(atoms, mol=ref_mol)
    assert match == ((0, 4, 5, 6),)

    # check everything else raises TypeError
    with pytest.raises(TypeError):
        rdkit2ase.match_substructure(atoms, 42)  # type: ignore[arg-type]


@pytest.mark.parametrize("packmol", ["packmol.jl"])
def test_match_substructur_box(packmol):
    atoms = rdkit2ase.smiles2conformers("CC(=O)O", 1)
    box = rdkit2ase.pack([atoms], counts=[3], packmol=packmol, density=0.5)

    # match CH3 fragment using smarts
    match = rdkit2ase.match_substructure(box, smarts="[C]([H])([H])[H]")
    assert match == ((0, 4, 5, 6), (8, 12, 13, 14), (16, 20, 21, 22))
    for m in match:
        assert box[m].get_chemical_symbols() == ["C", "H", "H", "H"]

    # now match using a ase.Atoms object
    ref = rdkit2ase.smiles2atoms("[C]([H])([H])[H]")
    match = rdkit2ase.match_substructure(box, fragment=ref)
    assert match == ((0, 4, 5, 6), (8, 12, 13, 14), (16, 20, 21, 22))

    # now match using a Chem.Mol object
    ref_mol = Chem.MolFromSmarts("[C]([H])([H])[H]")
    match = rdkit2ase.match_substructure(box, mol=ref_mol)
    assert match == ((0, 4, 5, 6), (8, 12, 13, 14), (16, 20, 21, 22))

    # check everything else raises TypeError
    with pytest.raises(TypeError):
        rdkit2ase.match_substructure(box, 42)


def test_get_substructure():
    atoms = rdkit2ase.smiles2atoms("C(C(CO[N+](=O)[O-])O[N+](=O)[O-])O[N+](=O)[O-]")
    # match NO3 group using smarts
    frames = rdkit2ase.get_substructures(atoms, smarts="[N+](=O)[O-]")
    assert len(frames) == 3
    for frame in frames:
        assert frame.get_chemical_symbols() == ["N", "O", "O"]

    # match using a ase.Atoms object
    ref = rdkit2ase.smiles2atoms("[N+](=O)[O-]")
    frames = rdkit2ase.get_substructures(atoms, fragment=ref)
    assert len(frames) == 3
    for frame in frames:
        assert frame.get_chemical_symbols() == ["N", "O", "O"]

    # match using a Chem.Mol object
    ref_mol = Chem.MolFromSmarts("[N+](=O)[O-]")
    frames = rdkit2ase.get_substructures(atoms, mol=ref_mol)
    assert len(frames) == 3
    for frame in frames:
        assert frame.get_chemical_symbols() == ["N", "O", "O"]


@pytest.mark.parametrize("packmol", ["packmol.jl"])
def test_get_substructure_box(packmol):
    atoms = rdkit2ase.smiles2conformers(
        "C(C(CO[N+](=O)[O-])O[N+](=O)[O-])O[N+](=O)[O-]", 1
    )
    box = rdkit2ase.pack([atoms], counts=[3], packmol=packmol, density=0.5)

    # match NO3 group using smarts
    frames = rdkit2ase.get_substructures(box, smarts="[N+](=O)[O-]")
    assert len(frames) == 9
    for frame in frames:
        assert frame.get_chemical_symbols() == ["N", "O", "O"]

    # match using a ase.Atoms object
    ref = rdkit2ase.smiles2atoms("[N+](=O)[O-]")
    frames = rdkit2ase.get_substructures(box, fragment=ref)
    assert len(frames) == 9
    for frame in frames:
        assert frame.get_chemical_symbols() == ["N", "O", "O"]

    # match using a Chem.Mol object
    ref_mol = Chem.MolFromSmarts("[N+](=O)[O-]")
    frames = rdkit2ase.get_substructures(box, mol=ref_mol)
    assert len(frames) == 9
    for frame in frames:
        assert frame.get_chemical_symbols() == ["N", "O", "O"]


@pytest.mark.parametrize("remove_connectivity", [True, False])
@pytest.mark.parametrize("packmol", ["packmol.jl"])
def test_iter_fragments(packmol, remove_connectivity):
    water = rdkit2ase.smiles2conformers("O", 1)
    box = rdkit2ase.pack([water], [10], density=500, packmol=packmol)
    if remove_connectivity:
        del box.info["connectivity"]
    fragments = list(rdkit2ase.iter_fragments(box))
    assert len(fragments) == 10
    for atoms in fragments:
        assert len(atoms) == 3


def test_bmim_bf4_no_info():
    bmim = rdkit2ase.smiles2conformers("CCCCN1C=C[N+](=C1)C", numConfs=1)
    bf4 = rdkit2ase.smiles2conformers("[B-](F)(F)(F)F", numConfs=1)

    box = rdkit2ase.pack(
        [bmim, bf4],
        counts=[10, 10],
        density=500,
        packmol="packmol.jl",
        tolerance=3,
    )
    del box.info["connectivity"]
    bf4_matches = rdkit2ase.match_substructure(
        box, smiles="[B-](F)(F)(F)F", suggestions=[]
    )
    assert len(bf4_matches) == 10
    for match in bf4_matches:
        assert box[match].get_chemical_symbols() == bf4[0].get_chemical_symbols()

    bmim_matches = rdkit2ase.match_substructure(
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

    bmim_matches = rdkit2ase.match_substructure(
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
    indices = rdkit2ase.select_atoms_flat_unique(ethanol_mol, "[#6]")
    assert sorted(indices) == [0, 1]


def test_select_oxygen(ethanol_mol):
    """Test selecting the oxygen atom."""
    indices = rdkit2ase.select_atoms_flat_unique(ethanol_mol, "[#8]")
    assert sorted(indices) == [2]


def test_no_matches(ethanol_mol):
    """Test a SMARTS pattern that has no matches."""
    indices = rdkit2ase.select_atoms_flat_unique(ethanol_mol, "[F]")  # Fluorine
    assert indices == []


# --- Hydrogen Handling Tests ---


def test_hydrogens_excluded_by_default(ethanol_mol):
    """Test that hydrogens are excluded by default."""
    # C-O bond involves atoms 1 and 2. Hydrogens attached are not included.
    indices = rdkit2ase.select_atoms_flat_unique(ethanol_mol, "CO")
    assert sorted(indices) == [1, 2]


def test_hydrogens_included(ethanol_mol):
    """Test the 'include' option for hydrogens."""
    # C-O bond (atoms 1, 2) plus attached hydrogens (atoms 6, 7, 8)
    # H on O is atom 8. Hs on C(1) are 6, 7.
    indices = rdkit2ase.select_atoms_flat_unique(ethanol_mol, "CO", hydrogens="include")
    # Expected: C(1), O(2), H(6), H(7), H(8)
    assert sorted(indices) == [1, 2, 6, 7, 8]


def test_hydrogens_isolated(ethanol_mol):
    """Test the 'isolated' option for hydrogens."""
    # Select ONLY the hydrogens from the C-O match
    indices = rdkit2ase.select_atoms_flat_unique(
        ethanol_mol, "CO", hydrogens="isolated"
    )
    # Expected: H(6), H(7), H(8)
    assert sorted(indices) == [6, 7, 8]


def test_smarts_with_explicit_hydrogens(ethanol_mol):
    """Test a SMARTS pattern that explicitly includes hydrogens."""
    # Find all hydrogens attached to an oxygen
    indices = rdkit2ase.select_atoms_flat_unique(
        ethanol_mol, "[#8]-[H]", hydrogens="include"
    )
    # Expected: O(2), H(8)
    assert sorted(indices) == [2, 8]

    # Now isolate only the hydrogen from that match
    h_indices = rdkit2ase.select_atoms_flat_unique(
        ethanol_mol, "[#8]-[H]", hydrogens="isolated"
    )
    assert sorted(h_indices) == [8]


# --- Mapped SMILES Tests ---


def test_mapped_smiles(ethanol_mol):
    """Test selecting only mapped atoms using a mapped SMILES pattern."""
    # The pattern "[C:1][C:2]O" matches atoms 0, 1, and 2,
    # but only C:1 and C:2 are mapped.
    # The function should now return only the indices of the mapped atoms.
    indices = rdkit2ase.select_atoms_flat_unique(ethanol_mol, "[C:1][C:2]O")
    # FIX: The test's expectation is updated to only expect the mapped carbons [0, 1].
    assert sorted(indices) == [0, 1]


def test_mapped_smiles_with_hydrogens(ethanol_mol):
    """Test mapped SMILES with hydrogen filtering."""
    # Pattern "C[O:1]" matches atoms C(1) and O(2), but only O(2) is mapped.
    # The core selection will be just atom 2.

    # Include hydrogens attached to the mapped oxygen
    indices_included = rdkit2ase.select_atoms_flat_unique(
        ethanol_mol, "C[O:1]", hydrogens="include"
    )
    # Expected: O(2) and its hydrogen H(8)
    assert sorted(indices_included) == [2, 8]

    # Exclude hydrogens (returns just the mapped heavy atom)
    indices_excluded = rdkit2ase.select_atoms_flat_unique(
        ethanol_mol, "C[O:1]", hydrogens="exclude"
    )
    assert sorted(indices_excluded) == [2]

    # Isolate only hydrogens attached to the mapped oxygen
    indices_isolated = rdkit2ase.select_atoms_flat_unique(
        ethanol_mol, "C[O:1]", hydrogens="isolated"
    )
    assert sorted(indices_isolated) == [8]


# --- Error Handling Tests ---


def test_invalid_smarts_raises_error():
    """Test that an invalid SMARTS string raises a ValueError."""
    mol = Chem.MolFromSmiles("C")
    with pytest.raises(ValueError, match="Invalid SMARTS/SMILES"):
        rdkit2ase.select_atoms_flat_unique(mol, "this is not valid")


# =============================================================================
# Test Cases for visualize_selected_molecules
# =============================================================================


def test_visualize_selected_molecules_basic(ethanol_mol):
    """Test basic visualization functionality."""
    # Select some atoms to highlight
    a = [0, 1]  # Carbons
    b = [2]  # Oxygen

    img = rdkit2ase.visualize_selected_molecules(ethanol_mol, a, b)
    assert img is not None


def test_visualize_selected_molecules_empty_selections(ethanol_mol):
    """Test visualization with empty selections."""
    img = rdkit2ase.visualize_selected_molecules(ethanol_mol, [], [])
    assert img is None


def test_visualize_selected_molecules_overlapping_selections(ethanol_mol):
    """Test visualization with overlapping selections (b takes precedence)."""
    a = [0, 1, 2]  # All heavy atoms
    b = [2]  # Oxygen (should be blue, not pink)

    img = rdkit2ase.visualize_selected_molecules(ethanol_mol, a, b)
    assert img is not None


def test_select_atoms_grouped(ethanol_water):
    """Test selecting atoms from disconnected fragments."""
    # ethanol_water is an ase.Atoms object with 2 ethanol and 2 water molecules.
    # We need to convert it to an RDKit Mol object first.
    mol = rdkit2ase.ase2rdkit(ethanol_water)

    # Test 1: Select all carbon atoms.
    # Should find 2 groups (the two ethanol molecules), each with 2 carbons.
    indices_carbons = rdkit2ase.select_atoms_grouped(mol, "[#6]")
    assert indices_carbons == [[0, 1], [9, 10]]
    # Verify they are indeed carbons
    for group in indices_carbons:
        for idx in group:
            assert mol.GetAtomWithIdx(idx).GetAtomicNum() == 6

    # Test 2: Select all oxygen atoms.
    # Should find 4 groups (2 ethanol, 2 water), each with 1 oxygen.
    indices_oxygens = rdkit2ase.select_atoms_grouped(mol, "[#8]")
    assert indices_oxygens == [[2], [11], [18], [21]]
    # Verify they are indeed oxygens
    for group in indices_oxygens:
        for idx in group:
            assert mol.GetAtomWithIdx(idx).GetAtomicNum() == 8

    # Test 3: Select a non-existent atom.
    indices_fluorine = rdkit2ase.select_atoms_grouped(mol, "[F]")
    assert indices_fluorine == []

    # Test 4: Select hydroxyl group in ethanol, including hydrogens.
    indices_hydroxyl = rdkit2ase.select_atoms_grouped(mol, "[OH]", hydrogens="include")
    assert indices_hydroxyl == [[2, 8], [11, 17]]
    for group in indices_hydroxyl:
        symbols = [mol.GetAtomWithIdx(idx).GetSymbol() for idx in group]
        assert "O" in symbols
        assert "H" in symbols
