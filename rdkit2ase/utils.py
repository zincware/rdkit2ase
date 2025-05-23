import ase
import ase.units


def calculate_density(atoms: ase.Atoms) -> float:
    """Calculates the density of an ASE Atoms object in kg/m^3.

    The density is calculated as the total mass (in kg) divided by the volume (in m^3).
    """
    total_mass_kg = sum(atoms.get_masses()) * ase.units._amu  # amu -> kg
    volume_m3 = atoms.get_volume() * 1e-30  # Å^3 -> m^3
    density = total_mass_kg / volume_m3  # kg/m^3
    return density


def calculate_box_dimensions(images: list[ase.Atoms], density: float) -> list[float]:
    """Calculates the dimensions of the simulation box

    based on the molar volume and target density.
    """
    total_mass = sum(sum(atoms.get_masses()) for atoms in images)
    molar_volume = total_mass / density / 1000  # m^3 / mol
    volume_per_mol = molar_volume * ase.units.m**3 / ase.units.mol
    box_edge = volume_per_mol ** (1 / 3)
    return [box_edge] * 3


def iter_molecules(atoms: ase.Atoms) -> list[ase.Atoms]:
    """Iterate over the molecules in an ASE Atoms object."""
    if "connectivity" not in atoms.info:
        raise ValueError("No connectivity information found in atoms.info")
    import networkx as nx

    # connectivity is a list of tuples (i, j, bond_type)
    graph = nx.Graph()
    for i, j, bond_type in atoms.info["connectivity"]:
        graph.add_edge(i, j)
    for component in nx.connected_components(graph):
        yield atoms[list(component)]
