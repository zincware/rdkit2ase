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
