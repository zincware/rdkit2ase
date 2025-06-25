import networkx as nx
import numpy as np
import ase
from ase.neighborlist import natural_cutoffs
from rdkit import Chem


def _suggestions2networkx(smiles: list[str]) -> list[nx.Graph]:
    from rdkit2ase import rdkit2networkx
    mols = []
    for _smiles in smiles:
        mol = Chem.MolFromSmiles(_smiles)
        mol = Chem.AddHs(mol)
    return [rdkit2networkx(mol) for mol in mols]

def update_bond_order(graph: nx.Graph, suggestions: list[str]|None = None) -> None:
    from rdkit2ase.bond_order import update_bond_order_from_suggestions, update_bond_order_determine, has_bond_order
    if not has_bond_order(graph):
        if suggestions is not None:
            suggestion_graphs = _suggestions2networkx(suggestions)
            update_bond_order_from_suggestions(graph, suggestion_graphs)
        update_bond_order_determine(graph)

def ase2networkx(atoms: ase.Atoms, suggestions: list[str]|None = None) -> nx.Graph:
    """
    Convert an ASE Atoms object to a NetworkX graph based on interatomic distances.

    Connectivity is determined using the sum of natural atomic cutoffs
    (scaled by 1.2). Each atom is represented as a node, and an edge is
    added if two atoms are within bonding distance.

    Parameters
    ----------
    atoms : ase.Atoms
        The ASE Atoms object to convert into a graph.

    Returns
    -------
    networkx.Graph
        An undirected NetworkX graph where:

        - **Nodes** represent atoms and include:

          - ``position`` (numpy.ndarray): Cartesian coordinates of the atom.
          - ``atomic_number`` (int): Atomic number.
          - ``original_index`` (int): Index in the original ASE Atoms object.
          - ``charge`` (int): Formal charge of the atom.

        - **Edges** represent interatomic bonds based on cutoff distance.
    """
    charges = atoms.get_initial_charges()

    if "connectivity" in atoms.info:
        connectivity = atoms.info["connectivity"]
        graph = nx.Graph()
        for i, j, bond_order in connectivity:
            graph.add_edge(
                i,
                j,
                bond_order=bond_order,
            )
        for i, atom in enumerate(atoms):
            graph.nodes[i]["position"] = atom.position
            graph.nodes[i]["atomic_number"] = atom.number
            graph.nodes[i]["original_index"] = atom.index
            graph.nodes[i]["charge"] = charges[i]
        return graph
    
    r_ij = atoms.get_all_distances(mic=True, vector=False)  # Get scalar distances

    atom_radii = np.array(natural_cutoffs(atoms, mult=1.2))

    pairwise_cutoffs = atom_radii[:, None] + atom_radii[None, :]

    connectivity_matrix = np.zeros((len(atoms), len(atoms)), dtype=int)

    d_ij_temp = r_ij.copy()
    np.fill_diagonal(d_ij_temp, np.inf)

    connectivity_matrix[d_ij_temp <= pairwise_cutoffs] = 1

    graph = nx.from_numpy_array(connectivity_matrix, edge_attr=None)
    for u, v in graph.edges():
        graph.edges[u, v]["bond_order"] = None

    for i, atom in enumerate(atoms):
        graph.nodes[i]["position"] = atom.position
        graph.nodes[i]["atomic_number"] = atom.number
        graph.nodes[i]["original_index"] = atom.index
        graph.nodes[i]["charge"] = charges[i]
    
    if suggestions is not None:
        update_bond_order(graph, suggestions)

    return graph


def ase2rdkit(atoms: ase.Atoms, suggestions: list[str]|None = None) -> Chem.Mol:
    from rdkit2ase import ase2networkx, networkx2rdkit
    graph = ase2networkx(atoms, suggestions=suggestions)    
    return networkx2rdkit(graph)


