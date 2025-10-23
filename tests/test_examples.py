import pathlib

import ase.io
import networkx as nx

import molify

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

    # ase2networkx now only determines connectivity, not bond orders
    # To get bond orders with suggestions, pass through networkx2rdkit
    graph_a = molify.ase2networkx(atoms)
    from molify import networkx2rdkit, rdkit2networkx

    mol_a = networkx2rdkit(graph_a, suggestions=[VC, DMC])
    a = rdkit2networkx(mol_a)

    graph_b = molify.ase2networkx(wrapped_atoms)
    mol_b = networkx2rdkit(graph_b, suggestions=[VC, DMC])
    b = rdkit2networkx(mol_b)

    for node in a.nodes:
        assert a[node] == b[node]
    for node in b.nodes:
        assert a[node] == b[node]
    assert nx.is_isomorphic(a, b)
