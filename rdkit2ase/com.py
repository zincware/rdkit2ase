import ase
import networkx as nx


def get_center_of_mass(atoms: ase.Atoms, unwrap: bool = True, **kwargs) -> ase.Atoms:
    """Compute the center of mass for each molecule in an ASE Atoms object."""
    from rdkit2ase import ase2networkx, networkx2ase, unwrap_structures

    if unwrap:
        graph = ase2networkx(atoms, **kwargs)
        atoms = unwrap_structures(atoms)  # Ensure structures are unwrapped
    graph = ase2networkx(atoms, **kwargs)

    centers = []
    masses = []
    for component in nx.connected_components(graph):
        subgraph = graph.subgraph(component)
        molecule = networkx2ase(subgraph)
        center_of_mass = molecule.get_center_of_mass()
        centers.append(center_of_mass)
        masses.append(molecule.get_masses().sum())
    return ase.Atoms(
        positions=centers, masses=masses, cell=atoms.get_cell(), pbc=atoms.pbc
    )
