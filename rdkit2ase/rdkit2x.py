import ase
import networkx as nx
from rdkit import Chem
from rdkit.Chem import AllChem


def rdkit2ase(mol, seed: int = 42) -> ase.Atoms:
    """Convert an RDKit molecule to an ASE atoms object."""
    smiles = Chem.MolToSmiles(mol)
    mol = Chem.AddHs(mol)
    charges = [atom.GetFormalCharge() for atom in mol.GetAtoms()]
    AllChem.EmbedMolecule(mol, randomSeed=seed)

    atoms = ase.Atoms(
        positions=mol.GetConformer().GetPositions(),
        numbers=[atom.GetAtomicNum() for atom in mol.GetAtoms()],
    )
    atoms.info["smiles"] = smiles
    atoms.info["connectivity"] = []
    for bond in mol.GetBonds():
        a1 = bond.GetBeginAtomIdx()
        a2 = bond.GetEndAtomIdx()
        order = bond.GetBondTypeAsDouble()
        atoms.info["connectivity"].append((a1, a2, order))

    if any(charge != 0 for charge in charges):
        atoms.set_initial_charges(charges)
    return atoms


def rdkit2networkx(mol: Chem.Mol) -> nx.Graph:
    """
    Convert an RDKit molecule to a NetworkX graph.

    Each atom is represented as a node, with attributes such as atomic number,
    formal charge, and original RDKit atom index. Each bond is represented as
    an edge, with the bond order stored as an edge attribute.

    Parameters
    ----------
    mol : Chem.Mol
        RDKit molecule object to be converted.

    Returns
    -------
    nx.Graph
        An undirected NetworkX graph where:

        - **Nodes** represent atoms and include:

          - ``atomic_number`` (int): Atomic number of the atom.
          - ``original_index`` (int): Original RDKit atom index.
          - ``charge`` (int): Formal charge of the atom.

        - **Edges** represent chemical bonds and include:

          - ``bond_order`` (float): Bond order
            (1: single, 2: double, 3: triple, 1.5: aromatic).
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
