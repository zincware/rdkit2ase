import networkx as nx
import numpy as np
from ase import Atoms
from ase.neighborlist import natural_cutoffs
from rdkit import Chem


# Map float bond orders to RDKit bond types
def bond_type_from_order(order):
    if order == 1.0:
        return Chem.BondType.SINGLE
    elif order == 2.0:
        return Chem.BondType.DOUBLE
    elif order == 3.0:
        return Chem.BondType.TRIPLE
    elif order == 1.5:
        return Chem.BondType.AROMATIC
    else:
        raise ValueError(f"Unsupported bond order: {order}")


def ase2networkx(atoms: Atoms) -> nx.Graph:
    """
    Converts an ASE Atoms object to a NetworkX graph based on interatomic distances.

    Connectivity between atoms is determined using the sum
    of their natural cutoffs (scaled by 1.2). Each node in the resulting
    graph represents an atom and stores its position, atomic number, and original index.

    Parameters
    ----------
    atoms : ase.Atoms
        The ASE Atoms object to convert into a graph.

    Returns
    -------
    networkx.Graph
        An undirected graph where nodes correspond to atoms and edges
        represent connectivity based on cutoff distances.
        Node attributes include:
            - 'position': numpy.ndarray, the atomic position.
            - 'atomic_number': int, the atomic number.
            - 'original_index': int, the atom's index in the original Atoms object.
            - 'charge': int, the initial charge of the atom.
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

    graph = nx.from_numpy_array(connectivity_matrix)
    for u, v in graph.edges():
        graph.edges[u, v]["bond_order"] = None

    for i, atom in enumerate(atoms):
        graph.nodes[i]["position"] = atom.position
        graph.nodes[i]["atomic_number"] = atom.number
        graph.nodes[i]["original_index"] = atom.index
        graph.nodes[i]["charge"] = charges[i]
    return graph


def rdkit2networkx(mol: Chem.Mol) -> nx.Graph:
    """
    Each atom in the molecule is represented as a node in the graph,
    with node attributes including atomic number, original RDKit
    atom index, and formal charge. Each bond is represented as an
    edge, with the bond order as an edge attribute.

    Parameters
    ----------
    mol : Chem.Mol
        RDKit molecule object to be converted.

    Returns
    -------
    nx.Graph
        A NetworkX graph where nodes correspond to atoms and edges correspond to bonds.
        Node attributes:
            - atomic_number (int): Atomic number of the atom.
            - original_index (int): Original RDKit atom index.
            - charge (int): Formal charge of the atom.
        Edge attributes:
            - bond_order (float): Bond order
            (1 for single, 2 for double, 3 for triple, 1.5 for aromatic).
    """
    graph = nx.Graph()
    for atom in mol.GetAtoms():
        graph.add_node(
            atom.GetIdx(),
            atomic_number=atom.GetAtomicNum(),
            original_index=atom.GetIdx(),
            charge=atom.GetFormalCharge(),
        )

    for bond in mol.GetBonds():
        bond_type = bond.GetBondType()
        bond_order = 0
        if bond_type == Chem.BondType.SINGLE:
            bond_order = 1
        elif bond_type == Chem.BondType.DOUBLE:
            bond_order = 2
        elif bond_type == Chem.BondType.TRIPLE:
            bond_order = 3
        elif bond_type == Chem.BondType.AROMATIC:
            bond_order = 1.5

        graph.add_edge(
            bond.GetBeginAtomIdx(), bond.GetEndAtomIdx(), bond_order=bond_order
        )
    return graph


def networkx2rdkit(graph: nx.Graph) -> Chem.Mol:
    """
    Converts a NetworkX graph back to an RDKit molecule.

    Parameters
    ----------
    graph : nx.Graph
        The input graph with nodes representing atoms and edges representing bonds.

    Returns
    -------
    Chem.Mol
        An RDKit molecule object reconstructed from the graph.
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
            bond_type_from_order(int(bond_order)),
        )

    Chem.SanitizeMol(mol)

    return mol.GetMol()


def networkx2atoms(graph: nx.Graph) -> Atoms:
    """
    Converts a NetworkX graph to an ASE Atoms object.

    Parameters
    ----------
    graph : nx.Graph
        The input graph with nodes representing atoms and edges representing bonds.

    Returns
    -------
    ase.Atoms
        An ASE Atoms object reconstructed from the graph.
    """
    positions = np.array([graph.nodes[n]["position"] for n in graph.nodes])
    numbers = np.array([graph.nodes[n]["atomic_number"] for n in graph.nodes])
    charges = np.array([graph.nodes[n]["charge"] for n in graph.nodes])

    atoms = Atoms(
        positions=positions,
        numbers=numbers,
        charges=charges,
        pbc=False,
    )

    connectivity = []
    for u, v, data in graph.edges(data=True):
        bond_order = data["bond_order"]
        connectivity.append((u, v, bond_order))

    atoms.info["connectivity"] = connectivity

    return atoms


def _recursive_match_attempt(  # noqa: C901
    template_graph: nx.Graph,
    ase_graph: nx.Graph,
    mapping: dict[int, int],
    rev_mapping: dict[int, int],
) -> dict[int, int] | None:
    """
    Attempts to find a subgraph isomorphism between the template and ASE graphs.

    Parameters
    ----------
    template_graph : nx.Graph
        The template graph (from RDKit).
    ase_graph : nx.Graph
        The target graph (from ASE Atoms).
    mapping : dict[int, int]
        Current mapping from template node index to ASE node index.
    rev_mapping : dict[int, int]
        Current reverse mapping from ASE node index to template node index.

    Returns
    -------
    dict[int, int] or None
        A completed mapping if successful, otherwise None.
    """
    if len(mapping) == len(template_graph.nodes):
        for u_template, v_template in template_graph.edges():
            if not (u_template in mapping and v_template in mapping):
                return None

            u_ase, v_ase = mapping[u_template], mapping[v_template]
            if not ase_graph.has_edge(u_ase, v_ase):
                return None
        return mapping

    unmapped_template_nodes = sorted(set(template_graph.nodes()) - set(mapping.keys()))

    if not unmapped_template_nodes:
        return None

    next_template_node_to_map = unmapped_template_nodes[0]

    potential_ase_candidates = sorted(ase_graph.nodes())

    for ase_candidate_node in potential_ase_candidates:
        if ase_candidate_node in rev_mapping:
            continue

        if (
            ase_graph.nodes[ase_candidate_node]["atomic_number"]
            != template_graph.nodes[next_template_node_to_map]["atomic_number"]
        ):
            continue

        compatible = True
        for mapped_template_neighbor in template_graph.neighbors(
            next_template_node_to_map
        ):
            if mapped_template_neighbor in mapping:
                ase_mapped_neighbor = mapping[mapped_template_neighbor]
                if not ase_graph.has_edge(ase_candidate_node, ase_mapped_neighbor):
                    compatible = False
                    break
        if not compatible:
            continue

        for mapped_ase_neighbor in ase_graph.neighbors(ase_candidate_node):
            if mapped_ase_neighbor in rev_mapping:
                corresponding_template_node = rev_mapping[mapped_ase_neighbor]
                if not template_graph.has_edge(
                    next_template_node_to_map, corresponding_template_node
                ):
                    compatible = False
                    break
        if not compatible:
            continue

        mapping[next_template_node_to_map] = ase_candidate_node
        rev_mapping[ase_candidate_node] = next_template_node_to_map

        result = _recursive_match_attempt(
            template_graph, ase_graph, mapping, rev_mapping
        )
        if result is not None:
            return result

        del rev_mapping[ase_candidate_node]
        del mapping[next_template_node_to_map]

    return None


def reconstruct_bonds_from_template(atoms_obj: Atoms, smiles_template: str):  # noqa: C901
    """
    Reconstruct bond information and atomic charges for an ASE Atoms object
    using a SMILES template.

    Parameters
    ----------
    atoms_obj : ase.Atoms
        The ASE Atoms object.
    smiles_template : str
        A SMILES string for the template molecule.

    Returns
    -------
    bonds : list of tuple
        A list of tuples (ase_idx1, ase_idx2, bond_order) representing bonds.
        Indices are 0-based original indices from the ASE Atoms object.
    charges : list of int
        A list of integers representing the formal charge on each atom
        in the original ASE atoms_obj, derived from the template. Atoms
        not part of the template mapping will have a charge of 0.

    Raises
    ------
    ValueError
        If no valid mapping is found or if atom counts are inconsistent.
    """
    if not atoms_obj:
        raise ValueError("Input ASE Atoms object is empty.")

    ase_graph = ase2networkx(atoms_obj)
    if not ase_graph.nodes:
        raise ValueError("Could not generate graph from ASE Atoms object (no nodes).")

    mol_template = Chem.MolFromSmiles(smiles_template)
    if mol_template is None:
        raise ValueError(f"Could not parse SMILES string: {smiles_template}")

    mol_template = Chem.AddHs(mol_template)
    Chem.SanitizeMol(mol_template)
    template_graph = rdkit2networkx(mol_template)

    if not template_graph.nodes:
        raise ValueError("Could not generate graph from SMILES template (no nodes).")

    if len(template_graph.nodes) > len(ase_graph.nodes):
        raise ValueError(
            f"Template molecule ({len(template_graph.nodes)} atoms) is larger than "
            f"ASE structure ({len(ase_graph.nodes)} atoms)."
        )

    template_atom_counts = {}
    for _, data in template_graph.nodes(data=True):
        num = data["atomic_number"]
        template_atom_counts[num] = template_atom_counts.get(num, 0) + 1

    ase_atom_counts = {}
    for _, data in ase_graph.nodes(data=True):
        num = data["atomic_number"]
        ase_atom_counts[num] = ase_atom_counts.get(num, 0) + 1

    for atom_num, count in template_atom_counts.items():
        if ase_atom_counts.get(atom_num, 0) < count:
            raise ValueError(
                f"Template requires {count} atom(s) of type {atom_num}, "
                f"but ASE object only has {ase_atom_counts.get(atom_num, 0)}."
            )

    if not template_atom_counts:
        return [], [0] * len(atoms_obj)

    min_count = float("inf")
    min_atom_type = -1
    for atom_num, count in template_atom_counts.items():
        if count < min_count:
            min_count = count
            min_atom_type = atom_num

    start_template_nodes = [
        n
        for n, data in template_graph.nodes(data=True)
        if data["atomic_number"] == min_atom_type
    ]

    start_ase_nodes = [
        n
        for n, data in ase_graph.nodes(data=True)
        if data["atomic_number"] == min_atom_type
    ]

    for template_start_node_idx in start_template_nodes:
        for ase_start_node_idx in start_ase_nodes:
            initial_mapping = {template_start_node_idx: ase_start_node_idx}
            initial_rev_mapping = {ase_start_node_idx: template_start_node_idx}

            solution_mapping = _recursive_match_attempt(
                template_graph,
                ase_graph,
                initial_mapping.copy(),
                initial_rev_mapping.copy(),
            )

            if solution_mapping is not None:
                bonds = []
                for u_template, v_template, data_template in template_graph.edges(
                    data=True
                ):
                    if (
                        u_template not in solution_mapping
                        or v_template not in solution_mapping
                    ):
                        continue

                    ase_node_u_idx_in_graph = solution_mapping[u_template]
                    ase_node_v_idx_in_graph = solution_mapping[v_template]

                    original_ase_idx_u = ase_graph.nodes[ase_node_u_idx_in_graph][
                        "original_index"
                    ]
                    original_ase_idx_v = ase_graph.nodes[ase_node_v_idx_in_graph][
                        "original_index"
                    ]

                    bond_order = data_template["bond_order"]

                    bond_tuple = tuple(
                        sorted((original_ase_idx_u, original_ase_idx_v))
                    ) + (bond_order,)
                    bonds.append(bond_tuple)

                ase_atom_charges = [0] * len(atoms_obj)
                for (
                    template_node_idx,
                    ase_node_idx_in_graph,
                ) in solution_mapping.items():
                    original_ase_idx = ase_graph.nodes[ase_node_idx_in_graph][
                        "original_index"
                    ]
                    charge = template_graph.nodes[template_node_idx]["charge"]
                    ase_atom_charges[original_ase_idx] = charge

                return sorted(set(bonds)), ase_atom_charges

    raise ValueError(
        "Could not find a valid mapping between the template and the ASE structure."
    )
