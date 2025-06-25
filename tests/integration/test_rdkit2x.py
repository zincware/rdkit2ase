import pytest
from rdkit import Chem

import rdkit2ase

# Shared test cases
SMILES_LIST = [
    # Simple neutral molecules
    "O",  # Water
    "CC",  # Ethane
    "C1CCCCC1",  # Cyclohexane
    "C1=CC=CC=C1",  # Benzene
    "C1=CC=CC=C1O",  # Phenol
    # Simple anions/cations
    "[Li+]",  # Lithium ion
    "[Na+]",  # Sodium ion
    "[Cl-]",  # Chloride
    "[OH-]",  # Hydroxide
    "[NH4+]",  # Ammonium
    "[CH3-]",  # Methyl anion
    "[C-]#N",  # Cyanide anion"
    # Phosphate and sulfate groups
    "OP(=O)(O)O ",  # H3PO4
    "OP(=O)(O)[O-]",  # H2PO4-
    # "[O-]P(=O)(O)[O-]",  # HPO4 2-
    # "[O-]P(=O)([O-])[O-]",  # PO4 3-
    "OP(=O)=O",  # HPO3
    # "[O-]P(=O)=O",  # PO3 -
    "OS(=O)(=O)O",  # H2SO4
    "OS(=O)(=O)[O-]",  # HSO4-
    # "[O-]S(=O)(=O)[O-]",  # SO4 2-
    # "[O-]S(=O)(=O)([O-])",  # SO3 2-
    # Multiply charged ions
    # "[Fe+3]",            # Iron(III)
    # "[Fe++]",            # Iron(II) alternative syntax
    # "[O-2]",             # Oxide dianion
    # "[Mg+2]",            # Magnesium ion
    # "[Ca+2]",            # Calcium ion
    # Charged organic fragments
    "C[N+](C)(C)C",  # Tetramethylammonium
    # "[N-]=[N+]=[N-]",       # Azide ion
    "C1=[N+](C=CC=C1)[O-]",  # Nitrobenzene
    # Complex anions
    "F[B-](F)(F)F",  # Tetrafluoroborate
    "F[P-](F)(F)(F)(F)F",  # Hexafluorophosphate
    # "[O-]C(=O)C(=O)[O-]",  # Oxalate dianion
    # Zwitterions
    "C(C(=O)[O-])N",  # Glycine
    "C1=CC(=CC=C1)[N+](=O)[O-]",  # Nitrobenzene
    # Aromatic heterocycles
    "C1CCNCC1",
    # Polyaromatics
    "C1CCC2CCCCC2C1",  # Naphthalene
    "C1CCC2C(C1)CCC1CCCCC12",  # Phenanthrene
]


@pytest.mark.parametrize("smiles", SMILES_LIST)
def test_rdkit2ase(smiles):
    mol = Chem.MolFromSmiles(smiles)
    mol = Chem.AddHs(mol)
    atoms = rdkit2ase.rdkit2ase(mol)
    assert sorted(atoms.get_chemical_symbols()) == sorted(
        [atom.GetSymbol() for atom in mol.GetAtoms()]
    )
    assert len(atoms) == mol.GetNumAtoms()

    connectivity = atoms.info["connectivity"]
    for i, j, order in connectivity:
        bond = mol.GetBondBetweenAtoms(i, j)
        assert bond is not None
        assert bond.GetBondTypeAsDouble() == order


@pytest.mark.parametrize("smiles", SMILES_LIST)
def test_rdkit2networkx(smiles):
    mol = Chem.MolFromSmiles(smiles)
    mol = Chem.AddHs(mol)
    graph = rdkit2ase.rdkit2networkx(mol)

    for u, v, data in graph.edges(data=True):
        bond_order = data.get("bond_order")
        bond = mol.GetBondBetweenAtoms(u, v)
        assert bond_order is not None
        assert bond is not None
        assert bond.GetBondTypeAsDouble() == bond_order

    for node_id, attributes in graph.nodes(data=True):
        atomic_number = attributes.get("atomic_number")
        charge = attributes.get("charge", 0)
        atom = mol.GetAtomWithIdx(node_id)
        assert atomic_number == atom.GetAtomicNum()
        assert charge == atom.GetFormalCharge()
