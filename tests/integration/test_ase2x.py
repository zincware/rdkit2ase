from unittest.mock import patch

import ase
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


@pytest.fixture
def atoms_and_connectivity(request):
    smiles = request.param
    atoms = rdkit2ase.smiles2atoms(smiles)
    connectivity = atoms.info["connectivity"]
    return smiles, atoms, connectivity


@pytest.mark.parametrize("atoms_and_connectivity", SMILES_LIST, indirect=True)
def test_ase2networkx(atoms_and_connectivity):
    _, atoms, connectivity = atoms_and_connectivity
    graph = rdkit2ase.ase2networkx(atoms)
    for i, j, bond_order in connectivity:
        assert graph.has_edge(i, j)
        assert graph.edges[i, j]["bond_order"] == bond_order


@pytest.mark.parametrize("atoms_and_connectivity", SMILES_LIST, indirect=True)
def test_ase2networkx_no_connectivity(atoms_and_connectivity):
    _, atoms, connectivity = atoms_and_connectivity
    atoms.info.pop("connectivity")
    graph = rdkit2ase.ase2networkx(atoms)
    for i, j, bond_order in connectivity:
        assert graph.has_edge(i, j)
        assert graph.edges[i, j]["bond_order"] is None


@pytest.mark.parametrize("atoms_and_connectivity", SMILES_LIST, indirect=True)
def test_ase2networkx_guess_connectivity(atoms_and_connectivity):
    _, atoms, connectivity = atoms_and_connectivity
    atoms.info.pop("connectivity")
    graph = rdkit2ase.ase2networkx(atoms, suggestions=[])
    for i, j, bond_order in connectivity:
        assert graph.has_edge(i, j)
        assert graph.edges[i, j]["bond_order"] == bond_order


@pytest.mark.parametrize("atoms_and_connectivity", SMILES_LIST, indirect=True)
def test_ase2networkx_smiles_connectivity(atoms_and_connectivity):
    smiles, atoms, connectivity = atoms_and_connectivity
    atoms.info.pop("connectivity")
    graph = rdkit2ase.ase2networkx(atoms, suggestions=[smiles])
    for i, j, bond_order in connectivity:
        assert graph.has_edge(i, j)
        assert graph.edges[i, j]["bond_order"] == bond_order


@pytest.mark.parametrize("atoms_and_connectivity", SMILES_LIST, indirect=True)
def test_ase2rdkit(atoms_and_connectivity):
    smiles, atoms, connectivity = atoms_and_connectivity
    mol = rdkit2ase.ase2rdkit(atoms)
    assert Chem.MolToSmiles(mol, canonical=True) == Chem.MolToSmiles(
        Chem.AddHs(Chem.MolFromSmiles(smiles)), canonical=True
    )

    for bond in mol.GetBonds():
        a1 = bond.GetBeginAtomIdx()
        a2 = bond.GetEndAtomIdx()
        order = bond.GetBondTypeAsDouble()
        if (a1, a2, order) in connectivity:
            pass
        elif (a2, a1, order) in connectivity:
            pass
        else:
            raise AssertionError(
                f"Bond ({a1}, {a2}, {order}) not found in connectivity: {connectivity}"
            )


@pytest.mark.parametrize("atoms_and_connectivity", SMILES_LIST, indirect=True)
def test_ase2rdkit_no_connectivity(atoms_and_connectivity):
    smiles, atoms, connectivity = atoms_and_connectivity
    atoms.info.pop("connectivity")
    if len(connectivity) > 0:
        with pytest.raises(ValueError):
            rdkit2ase.ase2rdkit(atoms)
    else:
        mol = rdkit2ase.ase2rdkit(atoms)
        assert Chem.MolToSmiles(mol, canonical=True) == Chem.MolToSmiles(
            Chem.AddHs(Chem.MolFromSmiles(smiles)), canonical=True
        )


@pytest.mark.parametrize("atoms_and_connectivity", SMILES_LIST, indirect=True)
def test_ase2rdkit_guess_connectivity(atoms_and_connectivity):
    smiles, atoms, connectivity = atoms_and_connectivity
    atoms.info.pop("connectivity")
    mol = rdkit2ase.ase2rdkit(atoms, suggestions=[])
    assert Chem.MolToSmiles(mol, canonical=True) == Chem.MolToSmiles(
        Chem.AddHs(Chem.MolFromSmiles(smiles)), canonical=True
    )


@pytest.mark.parametrize("atoms_and_connectivity", SMILES_LIST, indirect=True)
def test_ase2rdkit_smiles_connectivity(atoms_and_connectivity):
    smiles, atoms, connectivity = atoms_and_connectivity
    atoms.info.pop("connectivity")
    mol = rdkit2ase.ase2rdkit(atoms, suggestions=[smiles])
    assert Chem.MolToSmiles(mol, canonical=True) == Chem.MolToSmiles(
        Chem.AddHs(Chem.MolFromSmiles(smiles)), canonical=True
    )


def test_ase2networkx_ec_emc_li_pf6(ec_emc_li_pf6):
    graph = rdkit2ase.ase2networkx(ec_emc_li_pf6)
    assert len(graph.nodes) == 3 * 7 + 3 * 1 + 15 * 12 + 10 * 8
    connectivity = ec_emc_li_pf6.info["connectivity"]
    for i, j, bond_order in connectivity:
        assert graph.has_edge(i, j)
        assert graph.edges[i, j]["bond_order"] == bond_order

    li_nodes = [d for _, d in graph.nodes(data=True) if d["atomic_number"] == 3]
    assert len(li_nodes) == 3
    for d in li_nodes:
        assert d["charge"] == 1


def test_ase2networkx_ec_emc_li_pf6_no_connectivity(ec_emc_li_pf6):
    connectivity = ec_emc_li_pf6.info.pop("connectivity")
    ec_emc_li_pf6.set_initial_charges()
    graph = rdkit2ase.ase2networkx(ec_emc_li_pf6)
    for i, j, bond_order in connectivity:
        assert graph.has_edge(i, j)
        assert graph.edges[i, j]["bond_order"] is None

    # There is no charge information if not using suggestions
    # li_nodes = [d for _, d in graph.nodes(data=True) if d["atomic_number"] == 3]
    # assert len(li_nodes) == 3
    # for d in li_nodes:
    #     assert d["charge"] == 1


def test_ase2networkx_ec_emc_li_pf6_guess_connectivity(ec_emc_li_pf6):
    connectivity = ec_emc_li_pf6.info.pop("connectivity")
    ec_emc_li_pf6.set_initial_charges()
    graph = rdkit2ase.ase2networkx(ec_emc_li_pf6, suggestions=[])
    for i, j, bond_order in connectivity:
        assert graph.has_edge(i, j)
        assert graph.edges[i, j]["bond_order"] == bond_order

    li_nodes = [d for _, d in graph.nodes(data=True) if d["atomic_number"] == 3]
    assert len(li_nodes) == 3
    for d in li_nodes:
        assert d["charge"] == 1


@pytest.mark.parametrize("smiles", ["[CH3]", "[O]", "C[O]"])
def test_radicals(smiles):
    atoms = rdkit2ase.smiles2atoms(smiles)
    graph = rdkit2ase.ase2networkx(atoms)

    mol = Chem.MolFromSmiles(smiles)
    for i, atom in enumerate(mol.GetAtoms()):
        assert graph.nodes[i]["charge"] == atom.GetFormalCharge()


@pytest.mark.parametrize(
    "smiles,expected_charge",
    [("[NH4+]", 1), ("[O-2]", -2), ("[Fe+3]", 3), ("C[N+](C)(C)C", 1)],
)
def test_formal_charges(smiles, expected_charge):
    atoms = rdkit2ase.smiles2atoms(smiles)
    total_charge = sum(atoms.get_initial_charges())
    assert total_charge == pytest.approx(expected_charge)


def test_ase2networkx_pbc_flag():
    """Test that pbc flag controls periodic boundary conditions in bond detection."""
    # Create a simple periodic system with atoms that would be bonded across PBC
    atoms = ase.Atoms(
        symbols=["H", "H"],
        positions=[[0.0, 0.0, 0.0], [2.9, 0.0, 0.0]],  # Close across PBC
        cell=[3.0, 3.0, 3.0],
        pbc=True,
    )

    # Remove any existing connectivity
    if "connectivity" in atoms.info:
        atoms.info.pop("connectivity")

    # Test with pbc=True (should find bond across periodic boundary)
    graph_pbc_true = rdkit2ase.ase2networkx(atoms, pbc=True)

    # Test with pbc=False (should not find bond across periodic boundary)
    graph_pbc_false = rdkit2ase.ase2networkx(atoms, pbc=False)

    # With pbc=True, atoms should be bonded (distance is 0.1 across PBC)
    # With pbc=False, atoms should not be bonded (distance is 2.9 > typical H-H cutoff)
    assert graph_pbc_true.has_edge(0, 1), (
        "With pbc=True, should find bond across periodic boundary"
    )
    assert not graph_pbc_false.has_edge(0, 1), (
        "With pbc=False, should not find bond across periodic boundary"
    )


@patch("rdkit2ase.ase2x.vesin", None)
def test_ase2networkx_vesin_none():
    """Test that ase2networkx works correctly when vesin is not available."""
    atoms = rdkit2ase.smiles2atoms("CC")  # Simple ethane molecule
    atoms.info.pop(
        "connectivity"
    )  # Remove connectivity to force neighbor list calculation

    # Should work without vesin and use ASE neighbor_list
    graph = rdkit2ase.ase2networkx(atoms)

    # Should have the expected number of nodes and at least one edge (C-C bond)
    assert len(graph.nodes) == len(atoms)
    assert len(graph.edges) >= 1


def test_ase2networkx_vesin_available():
    """Test that vesin is available and can be used."""
    try:
        import vesin  # noqa: F401

        vesin_available = True
    except ImportError:
        vesin_available = False

    atoms = rdkit2ase.smiles2atoms("CC")  # Simple ethane molecule
    atoms.info.pop(
        "connectivity"
    )  # Remove connectivity to force neighbor list calculation

    if vesin_available:
        # Should work with vesin installed
        graph = rdkit2ase.ase2networkx(atoms, pbc=True)
        assert len(graph.nodes) == len(atoms)
        assert len(graph.edges) >= 1
    else:
        pytest.skip("vesin not available in this environment")
