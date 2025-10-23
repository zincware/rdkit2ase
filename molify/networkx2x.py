import ase
import networkx as nx
import numpy as np
from rdkit import Chem

from molify.utils import bond_type_from_order


def networkx2ase(graph: nx.Graph) -> ase.Atoms:
    """Convert a NetworkX graph to an ASE Atoms object.

    Parameters
    ----------
    graph : networkx.Graph
        The molecular graph to convert. Node attributes must include:
            * position (numpy.ndarray): Cartesian coordinates
            * atomic_number (int): Element atomic number
            * charge (float, optional): Formal charge
        Edge attributes must include:
            * bond_order (float): Bond order
        Optional graph attributes:
            * pbc (bool): Periodic boundary conditions
            * cell (numpy.ndarray): Unit cell vectors

    Returns
    -------
    ase.Atoms
        The resulting Atoms object with:
            * Atomic positions and numbers
            * Initial charges if present in graph
            * Connectivity information stored in atoms.info
            * PBC and cell if present in graph

    Examples
    --------
    >>> import networkx as nx
    >>> from molify import networkx2ase
    >>> graph = nx.Graph()
    >>> graph.add_node(0, position=[0,0,0], atomic_number=1, charge=0)
    >>> graph.add_node(1, position=[1,0,0], atomic_number=1, charge=0)
    >>> graph.add_edge(0, 1, bond_order=1.0)
    >>> atoms = networkx2ase(graph)
    >>> len(atoms)
    2
    """
    # Create mapping from original node indices to new sequential indices
    node_mapping = {node: i for i, node in enumerate(graph.nodes)}

    positions = np.array([graph.nodes[n]["position"] for n in graph.nodes])
    numbers = np.array([graph.nodes[n]["atomic_number"] for n in graph.nodes])
    charges = np.array([graph.nodes[n]["charge"] for n in graph.nodes])

    atoms = ase.Atoms(
        positions=positions,
        numbers=numbers,
        charges=charges,
        pbc=graph.graph.get("pbc", False),
        cell=graph.graph.get("cell", None),
    )

    connectivity = []
    for u, v, data in graph.edges(data=True):
        bond_order = data["bond_order"]
        # Map original node indices to new sequential indices
        new_u = node_mapping[u]
        new_v = node_mapping[v]
        connectivity.append((new_u, new_v, bond_order))

    atoms.info["connectivity"] = connectivity

    return atoms


def networkx2rdkit(graph: nx.Graph, suggestions: list[str] | None = None) -> Chem.Mol:
    """Convert a NetworkX graph to an RDKit molecule.

    Automatically determines bond orders for edges with bond_order=None.

    Parameters
    ----------
    graph : networkx.Graph
        The molecular graph to convert

        Node attributes:
            * atomic_number (int): Element atomic number
            * charge (float, optional): Formal charge

        Edge attributes:
            * bond_order (float or None): Bond order. If None, will be determined
              automatically.

    suggestions : list[str], optional
        Optional SMILES/SMARTS patterns to help determine bond orders.
        Used when edges have bond_order=None.

    Returns
    -------
    rdkit.Chem.Mol
        The resulting RDKit molecule with:
            * Atoms and bonds from the graph
            * Formal charges if specified
            * All bond orders determined
            * Sanitized molecular structure

    Raises
    ------
    ValueError
        If nodes are missing atomic_number attribute, or if bond order
        determination fails for any edge.

    Examples
    --------
    >>> import networkx as nx
    >>> from molify import networkx2rdkit
    >>> graph = nx.Graph()
    >>> graph.add_node(0, atomic_number=6, charge=0)
    >>> graph.add_node(1, atomic_number=8, charge=0)
    >>> graph.add_edge(0, 1, bond_order=2.0)
    >>> mol = networkx2rdkit(graph)
    >>> mol.GetNumAtoms()
    2

    >>> # With automatic bond order determination
    >>> graph2 = nx.Graph()
    >>> graph2.add_node(0, atomic_number=6, charge=0, position=[0, 0, 0])
    >>> graph2.add_node(1, atomic_number=8, charge=0, position=[1.2, 0, 0])
    >>> graph2.add_edge(0, 1, bond_order=None)  # Will be determined
    >>> mol2 = networkx2rdkit(graph2)
    """
    from molify.bond_order import update_bond_order

    # Check if any edges have bond_order=None and determine them if needed
    has_none_bond_orders = any(
        data.get("bond_order") is None for u, v, data in graph.edges(data=True)
    )

    if has_none_bond_orders:
        # Make a copy to avoid modifying the input graph
        graph = graph.copy()
        update_bond_order(graph, suggestions)

        # Verify all bond orders are now determined
        for u, v, data in graph.edges(data=True):
            bond_order = data.get("bond_order")
            if bond_order is None:
                raise ValueError(
                    f"Failed to determine bond order for edge ({u}, {v}).\n"
                    f"This can happen with:\n"
                    f"  - Unusual bonding situations\n"
                    f"  - Disconnected atoms\n"
                    f"  - Invalid atomic coordinates\n\n"
                    f"Try:\n"
                    f"  1. Check atom positions are reasonable\n"
                    f"  2. Provide SMILES suggestions for complex molecules\n"
                    f"  3. Manually set connectivity with bond orders"
                )

    mol = Chem.RWMol()
    nx_to_rdkit_atom_map = {}

    for node_id, attributes in graph.nodes(data=True):
        atomic_number = attributes.get("atomic_number")
        charge = attributes.get("charge", 0)

        if atomic_number is None:
            raise ValueError(f"Node {node_id} is missing 'atomic_number' attribute.")

        atom = Chem.Atom(int(atomic_number))
        atom.SetFormalCharge(int(charge))
        idx = mol.AddAtom(atom)
        nx_to_rdkit_atom_map[node_id] = idx

    for u, v, data in graph.edges(data=True):
        bond_order = data.get("bond_order")

        if bond_order is None:
            raise ValueError(f"Edge ({u}, {v}) is missing 'bond_order' attribute.")

        mol.AddBond(
            nx_to_rdkit_atom_map[u],
            nx_to_rdkit_atom_map[v],
            bond_type_from_order(bond_order),
        )

    Chem.SanitizeMol(mol)

    return mol.GetMol()
