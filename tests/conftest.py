import ase
import pytest

import molify


@pytest.fixture(scope="session")
def ethanol_water():
    ethanol = molify.smiles2conformers("CCO", numConfs=100)
    water = molify.smiles2conformers("O", numConfs=100)
    box = molify.pack(
        [ethanol, water], counts=[2, 2], density=700, packmol="packmol.jl"
    )
    return box.copy()


@pytest.fixture(scope="session")
def alanine_dipeptide() -> ase.Atoms:
    """Alanine dipeptide conformer using molify.smiles2conformers."""
    return molify.smiles2atoms("CC(=O)NC(C)C(=O)NC")


@pytest.fixture(scope="session")
def alanine_dipeptide_box(alanine_dipeptide) -> ase.Atoms:
    """Box of alanine dipeptide molecules using molify.pack"""
    box = molify.pack(
        [[alanine_dipeptide]], counts=[3], density=500, packmol="packmol.jl"
    )
    return box.copy()
