import numpy.testing as npt
import pytest
from rdkit import Chem

import molify


class SMILES:
    PF6: str = "F[P-](F)(F)(F)(F)F"
    Li: str = "[Li+]"
    EC: str = "C1COC(=O)O1"
    EMC: str = "CCOC(=O)OC"


@pytest.fixture
def ec_emc_li_pf6():
    atoms_pf6 = molify.smiles2conformers(SMILES.PF6, numConfs=10)
    atoms_li = molify.smiles2conformers(SMILES.Li, numConfs=10)
    atoms_ec = molify.smiles2conformers(SMILES.EC, numConfs=10)
    atoms_emc = molify.smiles2conformers(SMILES.EMC, numConfs=10)

    return molify.pack(
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
    atoms = molify.smiles2atoms(smiles)
    return molify.ase2networkx(atoms), smiles, atoms


@pytest.mark.parametrize("graph_smiles_atoms", SMILES_LIST, indirect=True)
def test_networkx2ase(graph_smiles_atoms):
    graph, smiles, atoms = graph_smiles_atoms

    new_atoms = molify.networkx2ase(graph)
    assert new_atoms == atoms

    new_connectivity = normalize_connectivity(new_atoms.info["connectivity"])
    old_connectivity = normalize_connectivity(atoms.info["connectivity"])

    assert new_connectivity == old_connectivity
    npt.assert_array_equal(new_atoms.positions, atoms.positions)
    npt.assert_array_equal(new_atoms.numbers, atoms.numbers)
    npt.assert_array_equal(new_atoms.get_initial_charges(), atoms.get_initial_charges())


def test_networkx2ase_ec_emc_li_pf6(ec_emc_li_pf6):
    atoms = ec_emc_li_pf6
    graph = molify.ase2networkx(atoms)

    new_atoms = molify.networkx2ase(graph)

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

    mol = molify.networkx2rdkit(graph)
    assert Chem.MolToSmiles(mol, canonical=True) == Chem.MolToSmiles(
        Chem.AddHs(Chem.MolFromSmiles(smiles)), canonical=True
    )


def test_networkx2rdkit_ec_emc_li_pf6(ec_emc_li_pf6):
    atoms = ec_emc_li_pf6
    graph = molify.ase2networkx(atoms)
    mol = molify.networkx2rdkit(graph)

    assert len(mol.GetAtoms()) == len(atoms)
    atom_types = [atom.GetAtomicNum() for atom in mol.GetAtoms()]
    assert sorted(atom_types) == sorted(atoms.get_atomic_numbers().tolist())

    connectivity = atoms.info["connectivity"]
    assert len(connectivity) == 266
    for i, j, bond_order in connectivity:
        bond = mol.GetBondBetweenAtoms(i, j)
        assert bond is not None
        assert bond.GetBondTypeAsDouble() == bond_order


def test_networkx2ase_subgraph_index_mapping():
    """Test that networkx2ase correctly maps node indices when working with subgraphs.

    This test ensures that when a subgraph is created with non-sequential node indices,
    the resulting ASE atoms object has correctly mapped connectivity information.
    """
    # Create a simple molecule and convert to graph
    atoms = molify.smiles2atoms("CCO")  # Ethanol
    graph = molify.ase2networkx(atoms)

    # Create a subgraph with non-sequential node indices
    # This simulates the scenario that caused the original bug
    subgraph_nodes = [1, 2, 4]  # Non-sequential indices
    subgraph = graph.subgraph(subgraph_nodes).copy()

    # Convert back to ASE atoms
    subgraph_atoms = molify.networkx2ase(subgraph)

    # Check that the atoms object has the correct number of atoms
    assert len(subgraph_atoms) == len(subgraph_nodes)

    # Check connectivity information
    connectivity = subgraph_atoms.info["connectivity"]

    # All indices in connectivity should be valid for the atoms object
    for i, j, bond_order in connectivity:
        assert 0 <= i < len(subgraph_atoms)
        assert 0 <= j < len(subgraph_atoms)
        assert isinstance(bond_order, float)

    # Test that unwrap_structures works correctly with this subgraph
    # (This was the original failing case)
    unwrapped = molify.unwrap_structures(subgraph_atoms)
    assert len(unwrapped) == len(subgraph_atoms)


# ============================================================================
# Tests for networkx2rdkit with None bond orders (new architecture)
# ============================================================================


@pytest.mark.parametrize("graph_smiles_atoms", SMILES_LIST, indirect=True)
def test_networkx2rdkit_with_none_bond_orders(graph_smiles_atoms):
    """Test networkx2rdkit determines bond orders when graph has bond_order=None."""
    graph, smiles, atoms = graph_smiles_atoms

    # Create a graph with bond_order=None by removing bond orders
    graph_no_orders = graph.copy()
    for u, v in graph_no_orders.edges():
        graph_no_orders.edges[u, v]["bond_order"] = None

    # This should determine bond orders automatically
    mol = molify.networkx2rdkit(graph_no_orders, suggestions=[])

    # Verify the molecule is correct
    assert Chem.MolToSmiles(mol, canonical=True) == Chem.MolToSmiles(
        Chem.AddHs(Chem.MolFromSmiles(smiles)), canonical=True
    )


@pytest.mark.parametrize("graph_smiles_atoms", SMILES_LIST, indirect=True)
def test_networkx2rdkit_with_smiles_suggestions(graph_smiles_atoms):
    """Test networkx2rdkit uses SMILES suggestions for bond order determination."""
    graph, smiles, atoms = graph_smiles_atoms

    # Create a graph with bond_order=None
    graph_no_orders = graph.copy()
    for u, v in graph_no_orders.edges():
        graph_no_orders.edges[u, v]["bond_order"] = None

    # Use SMILES suggestion for bond order determination
    mol = molify.networkx2rdkit(graph_no_orders, suggestions=[smiles])

    # Verify the molecule matches expected structure
    assert Chem.MolToSmiles(mol, canonical=True) == Chem.MolToSmiles(
        Chem.AddHs(Chem.MolFromSmiles(smiles)), canonical=True
    )


def test_networkx2rdkit_preserves_input_graph():
    """Test that networkx2rdkit doesn't modify the input graph."""
    atoms = molify.smiles2atoms("CC")
    graph = molify.ase2networkx(atoms)

    # Remove bond orders
    for u, v in graph.edges():
        graph.edges[u, v]["bond_order"] = None

    # Store original state
    original_edges = {(u, v): data.copy() for u, v, data in graph.edges(data=True)}

    # Call networkx2rdkit
    _ = molify.networkx2rdkit(graph, suggestions=[])

    # Verify original graph is unchanged
    for u, v, data in graph.edges(data=True):
        assert data["bond_order"] is None
        assert original_edges[(u, v)] == data


def test_networkx2rdkit_mixed_bond_orders():
    """Test networkx2rdkit with mixed None and specified bond orders."""
    # Create a simple molecule
    atoms = molify.smiles2atoms("CCO")
    graph = molify.ase2networkx(atoms)
    _ = atoms.info["connectivity"].copy()

    # Set some bond orders to None
    edges = list(graph.edges())
    if len(edges) >= 2:
        graph.edges[edges[0][0], edges[0][1]]["bond_order"] = None

    # This should determine only the None bond orders
    mol = molify.networkx2rdkit(graph, suggestions=[])

    # Verify molecule is correct
    expected_smiles = Chem.MolToSmiles(
        Chem.AddHs(Chem.MolFromSmiles("CCO")), canonical=True
    )
    assert Chem.MolToSmiles(mol, canonical=True) == expected_smiles


def test_networkx2rdkit_complex_molecule_with_none_bond_orders(ec_emc_li_pf6):
    """Test networkx2rdkit with complex molecule missing bond orders."""
    atoms = ec_emc_li_pf6
    connectivity = atoms.info["connectivity"].copy()
    graph = molify.ase2networkx(atoms)

    # Remove all bond orders
    for u, v in graph.edges():
        graph.edges[u, v]["bond_order"] = None

    # Determine bond orders
    mol = molify.networkx2rdkit(graph, suggestions=[])

    # Verify molecule has correct structure
    assert len(mol.GetAtoms()) == len(atoms)
    assert len(mol.GetBonds()) == len(connectivity)

    # Verify all bonds have determined orders
    for bond in mol.GetBonds():
        assert bond.GetBondTypeAsDouble() > 0


def test_networkx2rdkit_error_on_failed_determination():
    """Test networkx2rdkit raises error when bond determination fails."""
    import networkx as nx

    # Create a pathological graph with impossible geometry
    graph = nx.Graph()
    graph.add_node(0, atomic_number=6, charge=0, position=[0, 0, 0])
    graph.add_node(
        1, atomic_number=6, charge=0, position=[100, 100, 100]
    )  # Too far apart
    graph.add_edge(0, 1, bond_order=None)
    graph.graph["pbc"] = False
    graph.graph["cell"] = None

    # This should raise an error (from underlying bond determination)
    with pytest.raises(ValueError, match="Failed to determine bonds"):
        molify.networkx2rdkit(graph, suggestions=[])
