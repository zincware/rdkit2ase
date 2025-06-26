import ase
import networkx as nx
import numpy as np
from rdkit import Chem

from rdkit2ase.utils import bond_type_from_order


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
    >>> from rdkit2ase import networkx2ase
    >>> graph = nx.Graph()
    >>> graph.add_node(0, position=[0,0,0], atomic_number=1, charge=0)
    >>> graph.add_node(1, position=[1,0,0], atomic_number=1, charge=0)
    >>> graph.add_edge(0, 1, bond_order=1.0)
    >>> atoms = networkx2ase(graph)
    >>> len(atoms)
    2
    """
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
        connectivity.append((u, v, bond_order))

    atoms.info["connectivity"] = connectivity

    return atoms


def networkx2rdkit(graph: nx.Graph) -> Chem.Mol:
    """Convert a NetworkX graph to an RDKit molecule.

    Parameters
    ----------
    graph : networkx.Graph
        The molecular graph to convert

        Node attributes:
            * atomic_number (int): Element atomic number
            * charge (float, optional): Formal charge

        Edge attributes:
            * bond_order (float): Bond order

    Returns
    -------
    rdkit.Chem.Mol
        The resulting RDKit molecule with:
            * Atoms and bonds from the graph
            * Formal charges if specified
            * Sanitized molecular structure

    Raises
    ------
    ValueError:
        If nodes are missing atomic_number attribute,
        or if edges are missing bond_order attribute

    Examples
    --------
    >>> import networkx as nx
    >>> from rdkit2ase import networkx2rdkit
    >>> graph = nx.Graph()
    >>> graph.add_node(0, atomic_number=6, charge=0)
    >>> graph.add_node(1, atomic_number=8, charge=0)
    >>> graph.add_edge(0, 1, bond_order=2.0)
    >>> mol = networkx2rdkit(graph)
    >>> mol.GetNumAtoms()
    2
    """
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
