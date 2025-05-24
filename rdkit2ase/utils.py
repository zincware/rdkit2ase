from collections import defaultdict, deque

import ase
import ase.units


def find_connected_components(connectivity: list[tuple[int, int, float]]):
    try:
        import networkx as nx

        graph = nx.Graph()
        for i, j, _ in connectivity:
            graph.add_edge(i, j)
        for component in nx.connected_components(graph):
            yield component
    except ImportError:
        adjacency = defaultdict(list)
        for i, j, _ in connectivity:
            adjacency[i].append(j)
            adjacency[j].append(i)

        visited = set()
        for start in range(len(adjacency)):
            if start in visited or start not in adjacency:
                continue
            # BFS or DFS to collect connected component
            component = []
            queue = deque([start])
            while queue:
                node = queue.pop()
                if node in visited:
                    continue
                visited.add(node)
                component.append(node)
                for neighbor in adjacency[node]:
                    if neighbor not in visited:
                        queue.append(neighbor)
            yield component


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
