from collections import defaultdict

import ase
import ase.units
import numpy as np
from ase.data import covalent_radii
from ase.neighborlist import NeighborList


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
        for start in adjacency:
            if start in visited:
                continue

            component = []
            stack = [start]
            while stack:
                node = stack.pop()
                if node in visited:
                    continue
                visited.add(node)
                component.append(node)
                stack.extend(n for n in adjacency[node] if n not in visited)

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


def unwrap_molecule(atoms, scale=1.2):
    """Robust unwrapping across PBC using root-referenced traversal."""
    positions = atoms.get_positions()
    cell = atoms.get_cell()
    numbers = atoms.get_atomic_numbers()

    # Cutoffs from covalent radii
    radii = scale * covalent_radii[numbers]
    cutoffs = radii

    # Build PBC-aware neighbor list
    nl = NeighborList(cutoffs=cutoffs, skin=0.3, self_interaction=False, bothways=True)
    nl.update(atoms)

    n_atoms = len(atoms)
    visited = np.zeros(n_atoms, dtype=bool)
    image_shifts = np.zeros(
        (n_atoms, 3), dtype=int
    )  # Tracks image shifts (integer multiples of cell)

    def traverse_molecule(root):
        """Traverse and unwrap molecule starting from root atom."""
        stack = [root]
        visited[root] = True
        while stack:
            i = stack.pop()
            neighbors, offsets = nl.get_neighbors(i)
            for j, offset in zip(neighbors, offsets):
                if visited[j]:
                    continue
                image_shifts[j] = (
                    image_shifts[i] + offset
                )  # Accumulate total image shift
                visited[j] = True
                stack.append(j)

    # Unwrap all disconnected fragments
    for i in range(n_atoms):
        if not visited[i]:
            traverse_molecule(i)

    # Apply accumulated image shifts to get true positions
    shifts = np.dot(image_shifts, cell)
    new_positions = positions + shifts
    atoms.set_positions(new_positions)

    return atoms
