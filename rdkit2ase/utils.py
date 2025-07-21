import io
from collections import defaultdict

import ase.io
import ase.units
import networkx as nx
import numpy as np
import rdkit.Chem
import rdkit.Chem.rdDetermineBonds
from rdkit import Chem


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


def unwrap_structures(atoms, scale=1.2, **kwargs) -> ase.Atoms:
    """Unwrap molecular structures across periodic boundary conditions (PBC).

    This function corrects atomic positions that have been wrapped across periodic
    boundaries, ensuring that bonded atoms appear as continuous molecular structures.
    It can handle multiple disconnected molecules within the same unit cell.

    The algorithm works by:
    1. Building a connectivity graph based on covalent radii
    2. Traversing each connected component (molecule) using depth-first search
    3. Accumulating periodic image shifts to maintain molecular connectivity
    4. Applying the shifts to obtain unwrapped coordinates

    Parameters
    ----------
    atoms : ase.Atoms
        The ASE Atoms object containing wrapped atomic positions
    scale : float, optional
        Scale factor for covalent radii cutoffs used in bond detection.
        Larger values include more distant neighbors as bonded.
        Default is 1.2.
    **kwargs : dict
        Additional keyword arguments to pass to the `ase2networkx` function.

    Returns
    -------
    ase.Atoms
        A new ASE Atoms object with unwrapped atomic positions. The original
        atoms object is not modified.

    Notes
    -----
    - The function preserves the original atoms object and returns a copy
    - Works with any periodic boundary conditions (1D, 2D, or 3D)
    - Handles multiple disconnected molecules/fragments
    - Uses covalent radii scaled by the `scale` parameter for bond detection

    Examples
    --------
    >>> import numpy as np
    >>> from rdkit2ase import smiles2conformers, pack, unwrap_structures
    >>>
    >>> # Create a realistic molecular system
    >>> water = smiles2conformers("O", numConfs=1)
    >>> ethanol = smiles2conformers("CCO", numConfs=1)
    >>>
    >>> # Pack molecules into a periodic box
    >>> packed_system = pack(
    ...     data=[water, ethanol],
    ...     counts=[10, 5],
    ...     density=800
    ... )
    >>>
    >>> # Simulate wrapped coordinates (as might occur in MD)
    >>> cell = packed_system.get_cell()
    >>> positions = packed_system.get_positions()
    >>>
    >>> # Artificially move the box and wrap it again
    >>> positions += np.array([cell[0, 0] * 0.5, 0, 0])  # Shift by half box length
    >>> wrapped_atoms = packed_system.copy()
    >>> wrapped_atoms.set_positions(positions)
    >>> wrapped_atoms.wrap()  # Wrap back into PBC
    >>>
    >>> # Unwrap the structures to get continuous molecules
    >>> unwrapped_atoms = unwrap_structures(wrapped_atoms)
    """
    from rdkit2ase import ase2networkx

    atoms = atoms.copy()  # Work on a copy to avoid modifying the original

    graph = ase2networkx(atoms, scale=scale, **kwargs)
    positions = atoms.get_positions()
    cell = atoms.get_cell()

    for component in nx.connected_components(graph):
        if len(component) == 1:
            continue
        # start at the atom closest to the center of the box
        root = min(
            component, key=lambda i: np.linalg.norm(positions[i] - cell.diagonal() / 2)
        )
        # now do a bfs traversal to unwrap the molecule
        for i, j in nx.dfs_tree(graph, source=root).edges():
            # i has already been unwrapped
            # j is the neighbor that needs to be unwrapped
            offsets = cell.scaled_positions(positions[i] - positions[j])
            offsets = offsets.round().astype(int)
            positions[j] += offsets @ cell  # Unwrap j to the same image as i
    atoms.set_positions(positions)
    # TODO: include connectivity information
    return atoms


def rdkit_determine_bonds(unwrapped_atoms: ase.Atoms) -> rdkit.Chem.Mol:
    if len(unwrapped_atoms) == 0:
        raise ValueError("Cannot determine bonds for an empty structure.")
    with io.StringIO() as f:
        ase.io.write(f, unwrapped_atoms, format="xyz")
        f.seek(0)
        xyz = f.read()
        mol = rdkit.Chem.MolFromXYZBlock(xyz)
    if len(unwrapped_atoms) == 1:
        if unwrapped_atoms.get_chemical_symbols()[0] in ["Li", "Na", "K", "Rb", "Cs"]:
            return rdkit.Chem.MolFromSmiles(
                f"[{unwrapped_atoms.get_chemical_symbols()[0]}+]"
            )
        if unwrapped_atoms.get_chemical_symbols()[0] in ["Cl", "Br", "I", "F"]:
            return rdkit.Chem.MolFromSmiles(
                f"[{unwrapped_atoms.get_chemical_symbols()[0]}-]"
            )
    if len(unwrapped_atoms) == 7:
        if sorted(unwrapped_atoms.get_chemical_symbols()) == sorted(
            ["P", "F", "F", "F", "F", "F", "F"]
        ):
            return rdkit.Chem.MolFromSmiles("F[P-](F)(F)(F)(F)F")
    for charge in [0, 1, -1, 2, -2]:
        try:
            rdkit.Chem.rdDetermineBonds.DetermineBonds(
                mol,
                charge=int(sum(unwrapped_atoms.get_initial_charges())) + charge,
            )
            return mol
        except ValueError:
            pass
    else:
        raise ValueError(
            "Failed to determine bonds for sub-structure up to charge "
            f"{sum(unwrapped_atoms.get_initial_charges()) + charge} "
            f"and {unwrapped_atoms.get_chemical_symbols()}"
        )


def suggestions2networkx(smiles: list[str]) -> list[nx.Graph]:
    from rdkit2ase import rdkit2networkx

    mols = []
    for _smiles in smiles:
        mol = Chem.MolFromSmiles(_smiles)
        mol = Chem.AddHs(mol)
        mols.append(mol)
    return [rdkit2networkx(mol) for mol in mols]
