import numpy.testing as npt
import pytest
from rdkit import Chem

import rdkit2ase


class SMILES:
    PF6: str = "F[P-](F)(F)(F)(F)F"
    Li: str = "[Li+]"
    EC: str = "C1COC(=O)O1"
    EMC: str = "CCOC(=O)OC"


@pytest.fixture
def ec_emc_li_pf6():
    atoms_pf6 = rdkit2ase.smiles2conformers(SMILES.PF6, numConfs=10)
    atoms_li = rdkit2ase.smiles2conformers(SMILES.Li, numConfs=10)
    atoms_ec = rdkit2ase.smiles2conformers(SMILES.EC, numConfs=10)
    atoms_emc = rdkit2ase.smiles2conformers(SMILES.EMC, numConfs=10)

    return rdkit2ase.pack(
        data=[atoms_pf6, atoms_li, atoms_ec, atoms_emc],
        counts=[3, 3, 8, 12],
        density=1400,
        packmol="packmol.jl",
    )


# Shared test cases
SMILES_LIST = [
    # Simple neutral molecules
    "O",  # Water
    "CC",  # Ethane
    "C1CCCCC1",  # Cyclohexane
    "C1=CC=CC=C1",  # Benzene
    "C1=CC=CC=C1O",  # Phenol
    # Simple anions/cations
    "[Li+]",  # Lithium ion
    "[Na+]",  # Sodium ion
    "[Cl-]",  # Chloride
    "[OH-]",  # Hydroxide
    "[NH4+]",  # Ammonium
    "[CH3-]",  # Methyl anion
    "[C-]#N",  # Cyanide anion"
    # Phosphate and sulfate groups
    "OP(=O)(O)O ",  # H3PO4
    "OP(=O)(O)[O-]",  # H2PO4-
    # "[O-]P(=O)(O)[O-]",  # HPO4 2-
    # "[O-]P(=O)([O-])[O-]",  # PO4 3-
    "OP(=O)=O",  # HPO3
    # "[O-]P(=O)=O",  # PO3 -
    "OS(=O)(=O)O",  # H2SO4
    "OS(=O)(=O)[O-]",  # HSO4-
    # "[O-]S(=O)(=O)[O-]",  # SO4 2-
    # "[O-]S(=O)(=O)([O-])",  # SO3 2-
    # Multiply charged ions
    # "[Fe+3]",            # Iron(III)
    # "[Fe++]",            # Iron(II) alternative syntax
    # "[O-2]",             # Oxide dianion
    # "[Mg+2]",            # Magnesium ion
    # "[Ca+2]",            # Calcium ion
    # Charged organic fragments
    "C[N+](C)(C)C",  # Tetramethylammonium
    # "[N-]=[N+]=[N-]",       # Azide ion
    "C1=[N+](C=CC=C1)[O-]",  # Nitrobenzene
    # Complex anions
    "F[B-](F)(F)F",  # Tetrafluoroborate
    "F[P-](F)(F)(F)(F)F",  # Hexafluorophosphate
    # "[O-]C(=O)C(=O)[O-]",  # Oxalate dianion
    # Zwitterions
    "C(C(=O)[O-])N",  # Glycine
    "C1=CC(=CC=C1)[N+](=O)[O-]",  # Nitrobenzene
    # Aromatic heterocycles
    "C1CCNCC1",
    # Polyaromatics
    "C1CCC2CCCCC2C1",  # Naphthalene
    "C1CCC2C(C1)CCC1CCCCC12",  # Phenanthrene
]


def normalize_connectivity(connectivity):
    """Normalize bond tuples to ensure consistent comparison."""
    return sorted((min(i, j), max(i, j), order) for i, j, order in connectivity)


@pytest.fixture
def graph_smiles_atoms(request):
    smiles = request.param
    atoms = rdkit2ase.smiles2atoms(smiles)
    return rdkit2ase.ase2networkx(atoms), smiles, atoms


@pytest.mark.parametrize("graph_smiles_atoms", SMILES_LIST, indirect=True)
def test_networkx2ase(graph_smiles_atoms):
    graph, smiles, atoms = graph_smiles_atoms

    new_atoms = rdkit2ase.networkx2ase(graph)
    assert new_atoms == atoms

    new_connectivity = normalize_connectivity(new_atoms.info["connectivity"])
    old_connectivity = normalize_connectivity(atoms.info["connectivity"])

    assert new_connectivity == old_connectivity
    npt.assert_array_equal(new_atoms.positions, atoms.positions)
    npt.assert_array_equal(new_atoms.numbers, atoms.numbers)
    npt.assert_array_equal(new_atoms.get_initial_charges(), atoms.get_initial_charges())


def test_networkx2ase_ec_emc_li_pf6(ec_emc_li_pf6):
    atoms = ec_emc_li_pf6
    graph = rdkit2ase.ase2networkx(atoms)

    new_atoms = rdkit2ase.networkx2ase(graph)

    new_connectivity = normalize_connectivity(new_atoms.info["connectivity"])
    old_connectivity = normalize_connectivity(atoms.info["connectivity"])

    assert new_atoms == atoms

    assert new_connectivity == old_connectivity
    npt.assert_array_equal(new_atoms.positions, atoms.positions)
    npt.assert_array_equal(new_atoms.numbers, atoms.numbers)
    npt.assert_array_equal(new_atoms.get_initial_charges(), atoms.get_initial_charges())
    npt.assert_array_equal(new_atoms.pbc, atoms.pbc)
    npt.assert_array_equal(new_atoms.cell, atoms.cell)


@pytest.mark.parametrize("graph_smiles_atoms", SMILES_LIST, indirect=True)
def test_networkx2rdkit(graph_smiles_atoms):
    graph, smiles, atoms = graph_smiles_atoms

    mol = rdkit2ase.networkx2rdkit(graph)
    assert Chem.MolToSmiles(mol, canonical=True) == Chem.MolToSmiles(
        Chem.AddHs(Chem.MolFromSmiles(smiles)), canonical=True
    )


def test_networkx2rdkit_ec_emc_li_pf6(ec_emc_li_pf6):
    atoms = ec_emc_li_pf6
    graph = rdkit2ase.ase2networkx(atoms)
    mol = rdkit2ase.networkx2rdkit(graph)

    assert len(mol.GetAtoms()) == len(atoms)
    atom_types = [atom.GetAtomicNum() for atom in mol.GetAtoms()]
    assert sorted(atom_types) == sorted(atoms.get_atomic_numbers().tolist())

    connectivity = atoms.info["connectivity"]
    assert len(connectivity) == 266
    for i, j, bond_order in connectivity:
        bond = mol.GetBondBetweenAtoms(i, j)
        assert bond is not None
        assert bond.GetBondTypeAsDouble() == bond_order
