import rdkit2ase
from rdkit2ase.connectivity import reconstruct_bonds_from_template

import pytest

@pytest.mark.parametrize("smiles", ["CCC", "CCCCN1C=C[N+](=C1)C", "[B-](F)(F)(F)F", "C1=CC=CC=C1", "C1=CC=CC=C1O"])
def test_reconstruct_bonds_from_template(smiles):
    atoms = rdkit2ase.smiles2atoms(smiles)
    true_connectivity = atoms.info["connectivity"]
    true_charges = atoms.get_initial_charges()
    atoms.info.pop("connectivity", None)
    atoms.info.pop("smiles", None)

    connectivity, charges = reconstruct_bonds_from_template(atoms, smiles)

    for bond in connectivity:
        if bond in true_connectivity:
            continue
        elif (bond[1], bond[0], bond[2]) in true_connectivity:
            continue
        else:
            raise AssertionError(f"Bond {bond} not in {true_connectivity}.")

    assert charges == true_charges.tolist()
