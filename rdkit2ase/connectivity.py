import numpy as np
import networkx as nx
from ase import Atoms
from ase.neighborlist import natural_cutoffs # For atoms_to_graph
from rdkit import Chem
# from rdkit.Chem import AllChem # Not strictly needed for this implementation but often useful

def atoms_to_graph(atoms: Atoms):
    """
    Converts an ASE Atoms object to a NetworkX graph.
    Connectivity is determined using natural_cutoffs.
    Nodes store atomic number and original index.
    This function is based on the user's provided snippet.
    """
    r_ij = atoms.get_all_distances(mic=True, vector=False) # Get scalar distances
    
    atom_radii = np.array(natural_cutoffs(atoms, mult=1.2)) 
    
    pairwise_cutoffs = atom_radii[:, None] + atom_radii[None, :]
    
    connectivity_matrix = np.zeros((len(atoms), len(atoms)), dtype=int)
    
    d_ij_temp = r_ij.copy()
    np.fill_diagonal(d_ij_temp, np.inf)
    
    connectivity_matrix[d_ij_temp <= pairwise_cutoffs] = 1
    
    G = nx.from_numpy_array(connectivity_matrix)
    
    for i, atom in enumerate(atoms):
        G.nodes[i]["position"] = atom.position
        G.nodes[i]["atomic_number"] = atom.number
        G.nodes[i]["original_index"] = atom.index 
    return G

def rdkit_mol_to_graph(mol: Chem.Mol):
    """
    Converts an RDKit Mol object to a NetworkX graph.
    Nodes store atomic_number, original_rdkit_index, and formal_charge.
    Edges store bond_order.
    """
    G = nx.Graph()
    for atom in mol.GetAtoms():
        G.add_node(atom.GetIdx(), 
                   atomic_number=atom.GetAtomicNum(),
                   original_rdkit_index=atom.GetIdx(),
                   formal_charge=atom.GetFormalCharge()) # Store formal charge
    
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
        
        G.add_edge(bond.GetBeginAtomIdx(), 
                   bond.GetEndAtomIdx(), 
                   bond_order=bond_order)
    return G

def _recursive_match_attempt(G_template, G_ase, mapping, rev_mapping):
    """
    Attempts to find a subgraph isomorphism.
    
    Args:
        G_template: The template graph (from RDKit).
        G_ase: The target graph (from ASE Atoms).
        mapping: Current mapping from template node index to ASE node index.
        rev_mapping: Current reverse mapping from ASE node index to template node index.

    Returns:
        A completed mapping if successful, otherwise None.
    """
    if len(mapping) == len(G_template.nodes):
        for u_template, v_template in G_template.edges():
            if not (u_template in mapping and v_template in mapping):
                return None 
            
            u_ase, v_ase = mapping[u_template], mapping[v_template]
            if not G_ase.has_edge(u_ase, v_ase):
                return None 
        return mapping 

    unmapped_template_nodes = sorted(list(set(G_template.nodes()) - set(mapping.keys())))
    
    if not unmapped_template_nodes: 
        return None 
    
    next_template_node_to_map = unmapped_template_nodes[0]

    potential_ase_candidates = sorted(list(G_ase.nodes()))

    for ase_candidate_node in potential_ase_candidates:
        if ase_candidate_node in rev_mapping:
            continue

        if G_ase.nodes[ase_candidate_node]['atomic_number'] != \
           G_template.nodes[next_template_node_to_map]['atomic_number']:
            continue

        compatible = True
        for mapped_template_neighbor in G_template.neighbors(next_template_node_to_map):
            if mapped_template_neighbor in mapping:  
                ase_mapped_neighbor = mapping[mapped_template_neighbor]
                if not G_ase.has_edge(ase_candidate_node, ase_mapped_neighbor):
                    compatible = False
                    break
        if not compatible:
            continue
        
        for mapped_ase_neighbor in G_ase.neighbors(ase_candidate_node):
            if mapped_ase_neighbor in rev_mapping: 
                corresponding_template_node = rev_mapping[mapped_ase_neighbor]
                if not G_template.has_edge(next_template_node_to_map, corresponding_template_node):
                    compatible = False
                    break
        if not compatible:
            continue

        mapping[next_template_node_to_map] = ase_candidate_node
        rev_mapping[ase_candidate_node] = next_template_node_to_map

        result = _recursive_match_attempt(G_template, G_ase, mapping, rev_mapping)
        if result is not None:
            return result  

        del rev_mapping[ase_candidate_node]
        del mapping[next_template_node_to_map]

    return None

def reconstruct_bonds_from_template(atoms_obj: Atoms, smiles_template: str):
    """
    Reconstructs bond information and atomic charges for an ASE Atoms object 
    using a SMILES template.

    Args:
        atoms_obj: The ASE Atoms object.
        smiles_template: A SMILES string for the template molecule.

    Returns:
        A tuple: (bonds, charges)
        - bonds: A list of tuples (ase_idx1, ase_idx2, bond_order) representing bonds.
                 Indices are 0-based original indices from the ASE Atoms object.
        - charges: A list of integers representing the formal charge on each atom
                   in the original ASE atoms_obj, derived from the template. Atoms
                   not part of the template mapping will have a charge of 0.

    Raises:
        ValueError: If no valid mapping is found or if atoms counts are inconsistent.
    """
    if not atoms_obj:
        raise ValueError("Input ASE Atoms object is empty.")

    G_ase = atoms_to_graph(atoms_obj)
    if not G_ase.nodes:
         raise ValueError("Could not generate graph from ASE Atoms object (no nodes).")

    mol_template = Chem.MolFromSmiles(smiles_template)
    if mol_template is None:
        raise ValueError(f"Could not parse SMILES string: {smiles_template}")
    
    mol_template = Chem.AddHs(mol_template)
    # It's good practice to sanitize and perceive charges if not already explicit in SMILES
    Chem.SanitizeMol(mol_template) 
    G_template = rdkit_mol_to_graph(mol_template)

    if not G_template.nodes:
        raise ValueError("Could not generate graph from SMILES template (no nodes).")

    if len(G_template.nodes) > len(G_ase.nodes):
        raise ValueError(
            f"Template molecule ({len(G_template.nodes)} atoms) is larger than "
            f"ASE structure ({len(G_ase.nodes)} atoms)."
        )

    template_atom_counts = {}
    for _, data in G_template.nodes(data=True):
        num = data['atomic_number']
        template_atom_counts[num] = template_atom_counts.get(num, 0) + 1

    ase_atom_counts = {}
    for _, data in G_ase.nodes(data=True):
        num = data['atomic_number']
        ase_atom_counts[num] = ase_atom_counts.get(num, 0) + 1
    
    for atom_num, count in template_atom_counts.items():
        if ase_atom_counts.get(atom_num, 0) < count:
            raise ValueError(
                f"Template requires {count} atom(s) of type {atom_num}, "
                f"but ASE object only has {ase_atom_counts.get(atom_num, 0)}."
            )

    if not template_atom_counts: 
        return [], [0] * len(atoms_obj) # Return empty bonds and zero charges for all atoms

    min_count = float('inf')
    min_atom_type = -1
    for atom_num, count in template_atom_counts.items():
        if count < min_count:
            min_count = count
            min_atom_type = atom_num
    
    start_template_nodes = [
        n for n, data in G_template.nodes(data=True) 
        if data['atomic_number'] == min_atom_type
    ]
    
    start_ase_nodes = [
        n for n, data in G_ase.nodes(data=True)
        if data['atomic_number'] == min_atom_type
    ]

    for template_start_node_idx in start_template_nodes:
        for ase_start_node_idx in start_ase_nodes:
            initial_mapping = {template_start_node_idx: ase_start_node_idx}
            initial_rev_mapping = {ase_start_node_idx: template_start_node_idx}
            
            solution_mapping = _recursive_match_attempt(
                G_template, G_ase, 
                initial_mapping.copy(), 
                initial_rev_mapping.copy()
            )

            if solution_mapping is not None:
                bonds = []
                for u_template, v_template, data_template in G_template.edges(data=True):
                    if u_template not in solution_mapping or v_template not in solution_mapping:
                        continue

                    ase_node_u_idx_in_graph = solution_mapping[u_template]
                    ase_node_v_idx_in_graph = solution_mapping[v_template]
                    
                    original_ase_idx_u = G_ase.nodes[ase_node_u_idx_in_graph]['original_index']
                    original_ase_idx_v = G_ase.nodes[ase_node_v_idx_in_graph]['original_index']
                    
                    bond_order = data_template['bond_order']
                    
                    bond_tuple = tuple(sorted((original_ase_idx_u, original_ase_idx_v))) + (bond_order,)
                    bonds.append(bond_tuple)
                
                # Initialize charges for all atoms in the original ASE object to 0
                ase_atom_charges = [0] * len(atoms_obj)
                # Assign charges based on the template mapping
                for template_node_idx, ase_node_idx_in_graph in solution_mapping.items():
                    original_ase_idx = G_ase.nodes[ase_node_idx_in_graph]['original_index']
                    charge = G_template.nodes[template_node_idx]['formal_charge']
                    ase_atom_charges[original_ase_idx] = charge
                
                return sorted(list(set(bonds))), ase_atom_charges

    raise ValueError("Could not find a valid mapping between the template and the ASE structure.")

