import ase
import numpy as np
import numpy.testing as npt
import pytest
import rdkit
from rdkit import Chem

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
def nacn():
    positions = [
        (0.000, 0.000, 0.000),  # Sodium
        (0.000, 0.000, 1.000),  # Carbon
        (0.000, 0.000, 2.000),  # Nitrogen
    ]
    symbols = ["Na", "C", "N"]
    atoms = ase.Atoms(symbols, positions=positions)
    atoms.set_initial_charges([1, 0, -1])
    return atoms


@pytest.fixture
def methanol():
    return rdkit.Chem.MolFromSmiles("CO")


def test_rdkit2ase(methanol):
    ase_atoms = rdkit2ase(methanol)
    assert isinstance(ase_atoms, ase.Atoms)
    assert ase_atoms.get_chemical_formula() == "CH4O"
    assert ase_atoms.info["smiles"] == "CO"

    assert ase_atoms.info["connectivity"] == [
        (0, 1, 1.0),
        (0, 2, 1.0),
        (0, 3, 1.0),
        (0, 4, 1.0),
        (1, 5, 1.0),
    ]
    assert ase_atoms.get_atomic_numbers().tolist() == [6, 8, 1, 1, 1, 1]


def test_ase2rdkit(methane):
    rdkit_mol = ase2rdkit(methane)

    # assert the smiles string
    smiles = rdkit.Chem.MolToSmiles(rdkit_mol)
    assert smiles == "[H]C([H])([H])[H]"


# TODO: this is currently not working!
# def test_ase2rdkit_nacn(nacn):
#     rdkit_mol = ase2rdkit(nacn)
#     assert isinstance(rdkit_mol, rdkit.Chem.Mol)
#     assert rdkit.Chem.MolToSmiles(rdkit_mol) == "[Na+].[C-]#N"
#     assert rdkit_mol.GetNumAtoms() == 3
#     assert rdkit_mol.GetNumBonds() == 1
#     assert rdkit_mol.GetBondWithIdx(0).GetBondTypeAsDouble() == 3.0


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
    assert atoms.info["smiles"] == smiles
    assert "initial_charges" not in atoms.arrays


def test_seeded_smiles_to_atoms():
    atoms = smiles2atoms("CCO", seed=42)
    npt.assert_almost_equal(
        atoms.get_positions(),
        [
            [-0.9534551842756259, 0.047804216096192946, 0.04249937663911893],
            [0.48791868768310137, -0.32153931807429764, -0.18904973635963],
            [1.276977526346879, 0.3247836784511733, 0.7376732425087382],
            [-1.3570126123500137, 0.7339751954162012, -0.7332225613847279],
            [-1.593320940326173, -0.8625698312047074, 0.021759475665733226],
            [-1.0710312763466532, 0.5964393608027696, 1.0111627605054987],
            [0.7874060542693432, -0.1406086817099794, -1.235774323901826],
            [0.5900278787458259, -1.426109660842922, -0.04403048576653677],
            [1.8324898662533213, 1.0478250410655707, 0.38898225209363174],
        ],
    )
    # now test with different seeds
    assert not np.allclose(
        atoms.get_positions(), smiles2atoms("CCO", seed=43).get_positions()
    )


def test_rdkit2ase_nacn():
    # create sodium cyanide molecule
    mol = Chem.MolFromSmiles("[C-]#N.[Na+]")
    mol = Chem.AddHs(mol)
    atoms = rdkit2ase(mol)
    assert atoms.get_chemical_formula() == "CNNa"
    assert atoms.info["smiles"] == "[C-]#N.[Na+]"
    assert atoms.info["connectivity"] == [
        (0, 1, 3.0),
    ]
    assert atoms.get_initial_charges().tolist() == [-1, 0, 1]
    assert atoms.get_atomic_numbers().tolist() == [6, 7, 11]
