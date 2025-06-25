import ase
import networkx as nx
import numpy as np
from rdkit import Chem

from rdkit2ase.utils import bond_type_from_order


def networkx2ase(graph: nx.Graph) -> ase.Atoms:
    positions = np.array([graph.nodes[n]["position"] for n in graph.nodes])
    numbers = np.array([graph.nodes[n]["atomic_number"] for n in graph.nodes])
    charges = np.array([graph.nodes[n]["charge"] for n in graph.nodes])

    atoms = ase.Atoms(
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


def networkx2rdkit(graph: nx.Graph) -> Chem.Mol:
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
