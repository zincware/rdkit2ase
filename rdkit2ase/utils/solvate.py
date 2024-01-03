import ase.io
from rdkit import Chem
from ..rdkit2ase import rdkit2ase
from .smiles import smiles2atoms
import tempfile
import subprocess
import typing as t

import pathlib

OBJ_OR_STR = t.Union[str, Chem.rdchem.Mol, ase.Atoms]

OBJ_OR_STR_OR_LIST = t.Union[OBJ_OR_STR, t.List[t.Tuple[OBJ_OR_STR, float]]]


def _get_atoms(mol: OBJ_OR_STR) -> ase.Atoms:
    if isinstance(mol, str):
        mol = smiles2atoms(mol)
    elif isinstance(mol, Chem.rdchem.Mol):
        mol = rdkit2ase(mol)
    return mol


def _get_box_from_density(
    mol: t.List[t.Tuple[ase.Atoms, float]], density: float
) -> list[float]:
    """Get the box size from the molar volume.

    Attributes
    ----------
    mol : dict
        Dictionary of molecules and their counts.
    density: float
        Density of the system in kg/m^3.
    """

    data = [atoms for atoms, _ in mol]
    counts = [count for _, count in mol]

    molar_mass = sum(
        sum(atoms.get_masses()) * count for atoms, count in zip(data, counts)
    )
    molar_volume = molar_mass / density / 1000  # m^3 / mol

    # convert to particles / A^3
    volume = molar_volume * ase.units.m**3 / ase.units.mol

    box = [pow(volume, 1 / 3) for _ in range(3)]
    return box


def pack(mol: OBJ_OR_STR_OR_LIST, box_size=None, density=None, pbc=True, tolerance=0.5):
    if not isinstance(mol, list):
        mol = [(_get_atoms(mol), 1)]
    else:
        mol = [(_get_atoms(m), c) for m, c in mol]

    if density is not None and box_size is None:
        box_size = _get_box_from_density(mol, density)
    elif density is not None and box_size is not None:
        # compute fractions from mol keys and fill the box keeping the ratio
        raise NotImplementedError()
    elif density is None and box_size is not None:
        pass
    else:
        raise ValueError("Either density or box_size must be specified.")

    if pbc:
        target_box = [x - tolerance for x in box_size]
    else:
        target_box = box_size

    file = f"""
    tolerance {tolerance}
    filetype xyz
    output mixture.xyz
    """
    for idx, (_, count) in enumerate(mol):
        file += f"""
        structure {idx}.xyz
            filetype xyz
            number {count}
            inside box 0 0 0 {" ".join([f"{x:.4f}" for x in target_box])}
        end structure
        """

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = pathlib.Path(tmpdir)
        for idx, (atoms, _) in enumerate(mol):
            ase.io.write(tmpdir / f"{idx}.xyz", atoms)
        with open(tmpdir / "pack.inp", "w") as f:
            f.write(file)
        subprocess.run(["packmol < pack.inp"], cwd=tmpdir, shell=True)

        atoms = ase.io.read(tmpdir / "mixture.xyz")

    atoms.cell = box_size
    if pbc:
        atoms.pbc = True

    return atoms
