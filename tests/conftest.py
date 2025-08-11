import pytest
import ase

import rdkit2ase


@pytest.fixture(scope="session")
def ethanol_water():
    ethanol = rdkit2ase.smiles2conformers("CCO", numConfs=100)
    water = rdkit2ase.smiles2conformers("O", numConfs=100)
    box = rdkit2ase.pack(
        [ethanol, water], counts=[2, 2], density=700, packmol="packmol.jl"
    )
    return box.copy()


@pytest.fixture(scope="session")
def alanine_dipeptide() -> ase.Atoms:
    """Alanine dipeptide conformer using rdkit2ase.smiles2conformers."""
    return rdkit2ase.smiles2atoms("CC(=O)NC(C)C(=O)NC")



@pytest.fixture(scope="session")
def alanine_dipeptide_box(alanine_dipeptide) -> ase.Atoms:
    """Box of alanine dipeptide molecules using rdkit2ase.pack with packmol='packmol.jl'."""
    box = rdkit2ase.pack(
        [alanine_dipeptide], counts=[3], density=500, packmol="packmol.jl"
    )
    return box.copy()
