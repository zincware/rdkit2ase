import pathlib
import subprocess
import tempfile
import typing as t

import ase.io
import numpy as np
from ase.io.proteindatabank import write_proteindatabank
from rdkit import Chem

OBJ_OR_STR = t.Union[str, Chem.rdchem.Mol, ase.Atoms]

OBJ_OR_STR_OR_LIST = t.Union[OBJ_OR_STR, t.List[t.Tuple[OBJ_OR_STR, float]]]

FORMAT = t.Literal["pdb", "xyz"]


def _get_cell_vectors(images: list[ase.Atoms], density: float) -> list[float]:
    """Get the box size from the molar volume.

    Attributes
    ----------
    images : list[ase.Atoms]
        All the atoms that should be packed.
    density: float
        Density of the system in kg/m^3.
    """
    molar_mass = sum(sum(atoms.get_masses()) for atoms in images)
    molar_volume = molar_mass / density / 1000  # m^3 / mol

    # convert to particles / A^3
    volume = molar_volume * ase.units.m**3 / ase.units.mol

    box = [volume ** (1 / 3) for _ in range(3)]
    return box


def pack(
    data: list[list[ase.Atoms]],
    counts: list[int],
    density: float,
    seed: int = 42,
    tolerance: float = 2,
    verbose: bool = False,
    packmol: str = "packmol",
    pbc: bool = True,
    _format: FORMAT = "pdb",
) -> ase.Atoms:
    """
    Pack the given molecules into a box with the specified density.

    Parameters
    ----------
    data : list[list[ase.Atoms]]
        A list of lists of ASE Atoms objects representing the molecules to be packed.
    counts : list[int]
        A list of integers representing the number of each type of molecule.
    density : float
        The target density of the packed system in kg/m^3.
    seed : int, optional
        The random seed for reproducibility, by default 42.
    tolerance : float, optional
        The tolerance for the packing algorithm, by default 2.
    verbose : bool, optional
        If True, enables logging of the packing process, by default False.
    packmol : str, optional
        The path to the packmol executable, by default "packmol".
    pbc : bool, optional
        Ensure tolerance across periodic boundaries, by default True.
    _format : str, optional
        The file format used for communication with packmol, by default "pdb".
        WARNING: Do not use "xyz". This might cause issues and
        is only implemented for debugging purposes.

    Returns
    -------
    ase.Atoms
        An ASE Atoms object representing the packed system.

    Example
    -------
    >>> from rdkit2ase import pack, smiles2conformers
    >>> water = smiles2conformers("O", 1)
    >>> ethanol = smiles2conformers("CCO", 1)
    >>> density = 1000  # kg/m^3
    >>> packed_system = pack([water, ethanol], [7, 5], density)
    >>> print(packed_system)
    Atoms(symbols='C10H44O12', pbc=True, cell=[8.4, 8.4, 8.4])
    """
    rng = np.random.default_rng(seed)
    selected_idx: list[np.ndarray] = []

    for images, count in zip(data, counts):
        selected_idx.append(
            rng.choice(range(len(images)), count, replace=len(images) < count)
        )

    images = [
        [data[category][idx] for idx in indices]
        for category, indices in enumerate(selected_idx)
    ]
    images = sum(images, [])

    cell = _get_cell_vectors(images=images, density=density)

    file = f"""
tolerance {tolerance}
filetype {_format}
output mixture.{_format}
seed {seed}"""
    if pbc:
        file += f"""
pbc 0 0 0 {" ".join([f"{x:.6f}" for x in cell])}
"""
    for category, indices in enumerate(selected_idx):
        for idx in indices:
            file += f"""
structure struct_{category}_{idx}.{_format}
    filetype {_format}
    number 1
    inside box 0. 0. 0. {cell[0]} {cell[1]} {cell[2]}

end structure
"""
    if verbose:
        print(f"{packmol} < ")
        print(file)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = pathlib.Path(tmpdir)
        for category, indices in enumerate(selected_idx):
            for idx in set(indices):
                atoms = data[category][idx]
                if _format == "pdb":
                    write_proteindatabank(
                        tmpdir / f"struct_{category}_{idx}.pdb", atoms
                    )
                elif _format == "xyz":
                    ase.io.write(tmpdir / f"struct_{category}_{idx}.xyz", atoms)
        (tmpdir / "pack.inp").write_text(file)
        subprocess.run(
            f"{packmol} < pack.inp",
            cwd=tmpdir,
            shell=True,
            check=True,
            capture_output=not verbose,
        )
        atoms: ase.Atoms = ase.io.read(tmpdir / f"mixture.{_format}")

    atoms.cell = cell
    atoms.pbc = True
    return atoms
