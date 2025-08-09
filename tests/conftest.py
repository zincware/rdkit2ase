import pytest

import rdkit2ase


@pytest.fixture(scope="session")
def ethanol_water():
    ethanol = rdkit2ase.smiles2conformers("CCO", numConfs=100)
    water = rdkit2ase.smiles2conformers("O", numConfs=100)
    box = rdkit2ase.pack(
        [ethanol, water], counts=[2, 2], density=700, packmol="packmol.jl"
    )
    return box.copy()
