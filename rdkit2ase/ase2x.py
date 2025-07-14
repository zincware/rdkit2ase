import ase
import networkx as nx
import numpy as np
from ase.neighborlist import natural_cutoffs, neighbor_list
from rdkit import Chem

from rdkit2ase.bond_order import update_bond_order

try:
    import vesin
except ImportError:
    vesin = None


def ase2networkx(
    atoms: ase.Atoms, suggestions: list[str] | None = None, pbc: bool = True
) -> nx.Graph:
    """Convert an ASE Atoms object to a NetworkX graph with bonding information.

    Parameters
    ----------
    atoms : ase.Atoms
        The ASE Atoms object to convert into a graph.
    suggestions : list[str], optional
        List of SMILES patterns to suggest bond orders (default is None).
        If None, bond order is only determined from connectivity.
        If you provide an empty list, bond order will be determined
        using rdkit's bond order determination algorithm.
        If SMILES patterns are provided, they will be used to
        suggest bond orders first, and then
        rdkit's bond order determination algorithm will be used.
    pbc : bool, optional
        Whether to consider periodic boundary conditions when calculating
        distances (default is True). If False, only connections within
        the unit cell are considered.

    Returns
    -------
    networkx.Graph
        An undirected NetworkX graph with atomic properties and bonding information.

    Notes
    -----
    The graph contains the following information:

    - Nodes represent atoms with properties:
        * position: Cartesian coordinates (numpy.ndarray)
        * atomic_number: Element atomic number (int)
        * original_index: Index in original Atoms object (int)
        * charge: Formal charge (float)
    - Edges represent bonds with:
        * bond_order: Bond order (float or None if unknown)
    - Graph properties include:
        * pbc: Periodic boundary conditions
        * cell: Unit cell vectors

    Connectivity is determined by:

    1. Using explicit connectivity if present in atoms.info
    2. Otherwise using distance-based cutoffs and use SMILES patterns
    3. Use rdkit's bond order determination algorithm if no suggestions are provided.

    Examples
    --------
    >>> from rdkit2ase import ase2networkx, smiles2atoms
    >>> atoms = smiles2atoms(smiles="O")
    >>> graph = ase2networkx(atoms)
    >>> len(graph.nodes)
    3
    >>> len(graph.edges)
    2
    """
    charges = atoms.get_initial_charges()

    if "connectivity" in atoms.info:
        connectivity = atoms.info["connectivity"]
        graph = nx.Graph()

        graph.graph["pbc"] = atoms.pbc
        graph.graph["cell"] = atoms.cell

        for i, atom in enumerate(atoms):
            graph.add_node(
                i,
                position=atom.position,
                atomic_number=atom.number,
                original_index=atom.index,
                charge=charges[i],
            )

        for i, j, bond_order in connectivity:
            graph.add_edge(
                i,
                j,
                bond_order=bond_order,
            )
        return graph

    # non-bonding positive charged atoms / ions.
    non_bonding_atomic_numbers = {3, 11, 19, 37, 55, 87}

    atomic_numbers = atoms.get_atomic_numbers()
    excluded_mask = np.isin(atomic_numbers, list(non_bonding_atomic_numbers))

    atom_radii = np.array(natural_cutoffs(atoms, mult=1.2))
    pairwise_cutoffs = atom_radii[:, None] + atom_radii[None, :]

    max_cutoff = np.max(pairwise_cutoffs)

    if vesin is not None:
        i, j, d, s = vesin.ase_neighbor_list(
            "ijdS", atoms, cutoff=max_cutoff, self_interaction=False
        )
    else:
        i, j, d, s = neighbor_list(
            "ijdS", atoms, cutoff=max_cutoff, self_interaction=False
        )

    # If pbc=False, filter out bonds that cross periodic boundaries
    if not pbc:
        # Keep only bonds where all shift vectors are zero (no periodic wrapping)
        non_periodic_mask = np.all(s == 0, axis=1)
        i = i[non_periodic_mask]
        j = j[non_periodic_mask]
        d = d[non_periodic_mask]

    d_ij = np.full((len(atoms), len(atoms)), np.inf)
    d_ij[i, j] = d
    np.fill_diagonal(d_ij, 0.0)

    # mask out non-bonding atoms
    d_ij[excluded_mask, :] = np.inf
    d_ij[:, excluded_mask] = np.inf

    connectivity_matrix = np.zeros((len(atoms), len(atoms)), dtype=int)

    np.fill_diagonal(d_ij, np.inf)

    connectivity_matrix[d_ij <= pairwise_cutoffs] = 1

    graph = nx.from_numpy_array(connectivity_matrix, edge_attr=None)
    for u, v in graph.edges():
        graph.edges[u, v]["bond_order"] = None

    for i, atom in enumerate(atoms):
        graph.nodes[i]["position"] = atom.position
        graph.nodes[i]["atomic_number"] = atom.number
        graph.nodes[i]["original_index"] = atom.index
        graph.nodes[i]["charge"] = float(charges[i])
        if atom.number in non_bonding_atomic_numbers:
            graph.nodes[i]["charge"] = 1.0

    graph.graph["pbc"] = atoms.pbc
    graph.graph["cell"] = atoms.cell

    if suggestions is not None:
        update_bond_order(graph, suggestions)

    return graph


def ase2rdkit(atoms: ase.Atoms, suggestions: list[str] | None = None) -> Chem.Mol:
    """Convert an ASE Atoms object to an RDKit molecule.

    Parameters
    ----------
    atoms : ase.Atoms
        The ASE Atoms object to convert.
    suggestions : list[str], optional
        List of SMARTS patterns to suggest bond orders (default is None).

    Returns
    -------
    rdkit.Chem.Mol
        The resulting RDKit molecule with 3D coordinates.

    Notes
    -----
    This function first converts the Atoms object to a NetworkX graph using
    ase2networkx, then converts the graph to an RDKit molecule using
    networkx2rdkit.

    Examples
    --------
    >>> from rdkit2ase import ase2rdkit, smiles2atoms
    >>> atoms = smiles2atoms(smiles="C=O")
    >>> mol = ase2rdkit(atoms)
    >>> mol.GetNumAtoms()
    4
    """
    from rdkit2ase import ase2networkx, networkx2rdkit

    graph = ase2networkx(atoms, suggestions=suggestions)
    return networkx2rdkit(graph)
