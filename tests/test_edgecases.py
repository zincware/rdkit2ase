import ase
import networkx as nx
import numpy.testing as npt
from rdkit import Chem

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


def test_empty_atoms_ase2rdkit():
    atoms = ase.Atoms()

    mol = rdkit2ase.ase2rdkit(atoms)
    assert mol.GetNumAtoms() == 0


def test_empty_atoms_ase2networkx():
    atoms = ase.Atoms()

    graph = rdkit2ase.ase2networkx(atoms)
    assert len(graph.nodes) == 0
    assert len(graph.edges) == 0


def test_single_atom_ase2rdkit():
    atoms = ase.Atoms("C", positions=[[0, 0, 0]])

    mol = rdkit2ase.ase2rdkit(atoms)
    assert mol.GetNumAtoms() == 1
    assert mol.GetAtomWithIdx(0).GetSymbol() == "C"


def test_single_atom_ase2networkx():
    atoms = ase.Atoms("C", positions=[[0, 0, 0]])

    graph = rdkit2ase.ase2networkx(atoms)
    assert len(graph.nodes) == 1
    assert len(graph.edges) == 0
    assert graph.nodes[0]["atomic_number"] == 6
    npt.assert_array_equal(graph.nodes[0]["position"], [0, 0, 0])
    assert graph.nodes[0]["charge"] == 0.0


def test_empty_graph_networkx2x():
    graph = nx.Graph()

    mol = rdkit2ase.networkx2rdkit(graph)
    assert mol.GetNumAtoms() == 0

    atoms = rdkit2ase.networkx2ase(graph)
    assert len(atoms) == 0


def test_empty_mol_rdkit2x():
    mol = Chem.Mol()

    atoms = rdkit2ase.rdkit2ase(mol)
    assert len(atoms) == 0

    graph = rdkit2ase.rdkit2networkx(mol)
    assert len(graph.nodes) == 0
    assert len(graph.edges) == 0
