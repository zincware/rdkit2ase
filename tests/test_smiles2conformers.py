import itertools

import numpy as np

from rdkit2ase import smiles2conformers


def test_smiles2conformers():
    smiles = "CCO"
    conformers = smiles2conformers(smiles, numConfs=10)
    assert len(conformers) == 10
    for a, b in itertools.combinations(conformers, 2):
        assert np.sum(np.abs(a.positions - b.positions)) > 0.1
