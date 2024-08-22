import ase
import pytest
import rdkit
import numpy.testing as npt
import numpy as np

from rdkit2ase import ase2rdkit, rdkit2ase, smiles2atoms


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

def test_seeded_smiles_to_atoms():
    atoms = smiles2atoms("CCO", seed=42)
    npt.assert_almost_equal(atoms.get_positions(), [[-0.92537086,  0.07420817,  0.03283984],
       [ 0.51231816, -0.4191819 , -0.07431482],
       [ 1.3778212 ,  0.44937969,  0.60442827],
       [-1.02252523,  1.07307188, -0.44285695],
       [-1.60443695, -0.63678765, -0.48322232],
       [-1.22359694,  0.14724363,  1.10020568],
       [ 0.80577952, -0.5060158 , -1.14510451],
       [ 0.58521288, -1.42739618,  0.3853352 ],
       [ 1.49479822,  1.24547818,  0.0226896 ]]
    )
    # now test with different seeds
    assert not np.allclose(atoms.get_positions(), smiles2atoms("CCO", seed=43).get_positions())
