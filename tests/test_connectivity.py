import pytest
from rdkit.Chem import AddHs, MolFromSmiles, MolToSmiles

import rdkit2ase

# from rdkit2ase.connectivity import reconstruct_bonds_from_template


@pytest.mark.parametrize("remove_connectivity", [True, False])
def test_ase2networkx(remove_connectivity):
    atoms = rdkit2ase.smiles2atoms("CO")
    if remove_connectivity:
        atoms.info.pop("connectivity", None)
    graph = rdkit2ase.ase2networkx(atoms)
    assert graph.number_of_nodes() == 6
    assert graph.number_of_edges() == 5

    assert graph.nodes[0]["atomic_number"] == 6
    assert graph.nodes[0]["position"].tolist() == pytest.approx(
        [-0.37, 0.0, 0.0], abs=1e-2
    )
    assert graph.nodes[0]["original_index"] == 0
    assert graph.nodes[0]["charge"] == 0

    assert graph.edges[(0, 1)]["bond_order"] in [None, 1]


def test_rdkit2networkx():
    etoh = MolFromSmiles("CCO")
    etoh = AddHs(etoh)
    graph = rdkit2ase.rdkit2networkx(etoh)
    assert graph.number_of_nodes() == 9
    assert graph.number_of_edges() == 8

    assert graph.nodes[0]["atomic_number"] == 6
    assert graph.nodes[0]["original_index"] == 0
    assert graph.nodes[0]["charge"] == 0


def test_networkx2rdkit():
    atoms = rdkit2ase.smiles2atoms("CO")
    graph = rdkit2ase.ase2networkx(atoms)
    mol = rdkit2ase.networkx2rdkit(graph)
    assert mol.GetNumAtoms() == 6
    assert mol.GetNumBonds() == 5

    # SMILES representation (including hydrogens)
    assert MolToSmiles(mol) == "[H]OC([H])([H])[H]"


def test_networkx2ase():
    atoms = rdkit2ase.smiles2atoms("CO")
    graph = rdkit2ase.ase2networkx(atoms)
    new_atoms = rdkit2ase.networkx2ase(graph)
    assert new_atoms.get_chemical_symbols() == atoms.get_chemical_symbols()
    assert new_atoms.get_positions().tolist() == atoms.get_positions().tolist()
    assert new_atoms.info["connectivity"] == atoms.info["connectivity"]
    assert (
        new_atoms.get_initial_charges().tolist() == atoms.get_initial_charges().tolist()
    )
