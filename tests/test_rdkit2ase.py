import rdkit
import ase
import pytest

from rdkit2ase import rdkit2ase, ase2rdkit, smiles2atoms

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

@pytest.mark.parametrize("smiles, formula", [
    ("CCO", "C2H6O"),
    ("CC(=O)O", "C2H4O2"),
    ("C1=CC=CC=C1", "C6H6"),
    ("C1=CC=CC=C1C", "C7H8"),
    ("C1=CC=CC=C1CC", "C8H10"),
    ("C1=CC=CC=C1CCC", "C9H12"),
])
def test_smiles_to_atoms(smiles, formula):
    atoms = smiles2atoms(smiles)
    assert isinstance(atoms, ase.Atoms)
    assert atoms.get_chemical_formula() == formula