import pathlib

import ase.io
import networkx as nx

import rdkit2ase

DATA = pathlib.Path(__file__).parent / "data"

PF6 = "F[P-](F)(F)(F)(F)F"
Li = "[Li+]"
EC = "C1COC(=O)O1"
EMC = "CCOC(=O)OC"

VC = "C1=COC(=O)O1"
DMC = "COC(=O)OC"


def test_partial_suggestions():
    atoms = ase.io.read(DATA / "ec_emc.xyz")
    wrapped_atoms = atoms.copy()
    wrapped_atoms.positions += [5, 5, 5]
    wrapped_atoms.wrap()

    # it will work, if all suggestions are present
    a = rdkit2ase.ase2networkx(atoms, suggestions=[VC, DMC])

    b = rdkit2ase.ase2networkx(wrapped_atoms, suggestions=[VC, DMC])

    for node in a.nodes:
        assert a[node] == b[node]
    for node in b.nodes:
        assert a[node] == b[node]
    assert nx.is_isomorphic(a, b)
