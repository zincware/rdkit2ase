import numpy.testing as npt

import rdkit2ase


def test_na_cn():
    # create sodium cyanide molecule
    naplus = rdkit2ase.smiles2conformers("[Na+]", numConfs=10)
    cnminus = rdkit2ase.smiles2conformers("[C-]#N", numConfs=10)
    assert len(naplus) == 10
    assert len(cnminus) == 10
    for atom in naplus:
        assert len(atom) == 1
        assert atom.get_initial_charges() == [1]
    for atom in cnminus:
        assert len(atom) == 2
        npt.assert_array_equal(atom.get_initial_charges(), [-1, 0])
