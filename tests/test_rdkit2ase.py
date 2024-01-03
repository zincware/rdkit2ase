import rdkit
import ase
import pytest
from ase.build import molecule

from rdkit2ase import rdkit2ase, ase2rdkit, smiles2atoms, pack


@pytest.fixture
def methane():
    positions = [
        (0.000, 0.000, 0.000),  # Carbon
        (0.629, 0.629, 0.629),  # Hydrogen
        (-0.629, -0.629, 0.629),  # Hydrogen
        (-0.629, 0.629, -0.629),  # Hydrogen
        (0.629, -0.629, -0.629),  # Hydrogen
    ]
    return ase.Atoms("CH4", positions=positions)


@pytest.fixture
def methanol():
    return rdkit.Chem.MolFromSmiles("CO")


def test_rdkit2ase(methanol):
    ase_atoms = rdkit2ase(methanol)
    assert isinstance(ase_atoms, ase.Atoms)
    assert ase_atoms.get_chemical_formula() == "CH4O"


def test_ase2rdkit(methane):
    rdkit_mol = ase2rdkit(methane)

    # assert the smiles string
    smiles = rdkit.Chem.MolToSmiles(rdkit_mol)
    assert smiles == "[H]C([H])([H])[H]"


def test_ase2rdkit2ase(methane):
    mol = ase2rdkit(methane)
    atoms = rdkit2ase(mol)
    assert isinstance(atoms, ase.Atoms)
    assert atoms.get_chemical_formula() == "CH4"


def test_rdkit2ase2rdkit(methanol):
    atoms = rdkit2ase(methanol)
    mol = ase2rdkit(atoms)
    assert isinstance(mol, rdkit.Chem.Mol)
    assert rdkit.Chem.MolToSmiles(rdkit.Chem.RemoveHs(mol)) == "CO"


@pytest.mark.parametrize(
    "smiles, formula",
    [
        ("CCO", "C2H6O"),
        ("CC(=O)O", "C2H4O2"),
        ("C1=CC=CC=C1", "C6H6"),
        ("C1=CC=CC=C1C", "C7H8"),
        ("C1=CC=CC=C1CC", "C8H10"),
        ("C1=CC=CC=C1CCC", "C9H12"),
    ],
)
def test_smiles_to_atoms(smiles, formula):
    atoms = smiles2atoms(smiles)
    assert isinstance(atoms, ase.Atoms)
    assert atoms.get_chemical_formula() == formula


def test_pack_density():
    atoms = pack("CCO", density=1000)
    assert atoms.get_chemical_formula() == "C2H6O"

    atoms = pack([("CCO", 2)], density=1000)
    assert atoms.get_chemical_formula() == "C4H12O2"

    atoms = pack([("CCO", 2), ("O", 1), ("Cl", 3)], density=1000)
    assert atoms.get_chemical_formula() == "C4H17Cl3O3"


def test_pack_box():
    atoms = pack("CCO", box_size=[5, 5, 5], pbc=False)
    assert atoms.get_chemical_formula() == "C2H6O"
    assert atoms.get_volume() == pytest.approx(125)

    atoms = pack([("CCO", 1)], box_size=[5, 5, 5], pbc=True, tolerance=0)
    assert atoms.get_chemical_formula() == "C2H6O"
    assert atoms.get_volume() == pytest.approx(125)

    atoms = pack([("CCO", 1)], box_size=[5, 5, 5], pbc=True, tolerance=0.5)
    assert atoms.get_chemical_formula() == "C2H6O"
    assert atoms.get_volume() == pytest.approx(125)


def test_pack_atoms():
    atoms = pack([(molecule("CH4"), 2)], density=800)
    assert atoms.get_chemical_formula() == "C2H8"
    assert atoms.get_volume() == pytest.approx(66.6, abs=0.001)
