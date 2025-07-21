import ase
import networkx as nx


def get_centers_of_mass(atoms: ase.Atoms, unwrap: bool = True, **kwargs) -> ase.Atoms:
    """Compute the center of mass for each molecule in an ASE Atoms object.

    Parameters
    ----------
    atoms : ase.Atoms
        The ASE Atoms object containing the molecular structures.
    unwrap : bool, optional
        If True, unwrap the structures before computing the center of mass.
    **kwargs : dict
        Additional keyword arguments to pass to the `ase2networkx` function.

    Returns
    -------
    ase.Atoms
        An ASE Atoms object containing the centers of mass of each molecule,
        with the masses of the molecules as the `masses` attribute.

    Example
    -------
    >>> from rdkit2ase import smiles2conformers, pack, get_centers_of_mass
    >>> ethanol = smiles2conformers("CCO", numConfs=10)
    >>> box = pack([ethanol], [10], density=786)
    >>> centers = get_centers_of_mass(box)
    >>> print(centers.get_positions().shape)
    (10, 3)
    """
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
