import typing as tp

import ase
from ase.build import separate
from rdkit import Chem
from rdkit.Chem import Draw

from rdkit2ase.ase2x import ase2rdkit
from rdkit2ase.utils import find_connected_components


def match_substructure(
    atoms: ase.Atoms,
    smiles: str | None = None,
    smarts: str | None = None,
    mol: Chem.Mol | None = None,
    fragment: ase.Atoms | None = None,
    **kwargs,
) -> tuple[tuple[int, ...]]:
    """
    Find all matches of a substructure pattern in a given ASE Atoms object.

    Parameters
    ----------
    atoms : ase.Atoms
        The molecule or structure in which to search for substructure matches.
    smiles : str, optional
        A SMILES string representing the substructure pattern to match.
    smarts : str, optional
        A SMARTS string representing the substructure pattern to match.
    mol : Chem.Mol, optional
        An RDKit Mol object representing the substructure pattern to match.
    fragment : ase.Atoms, optional
        An ASE Atoms object representing the substructure pattern to match.
        If provided, it will be converted to an RDKit Mol object for matching.
    **kwargs
        Additional keyword arguments passed to `ase2rdkit`.

    Returns
    -------
    tuple of tuple of int
        A tuple of atom index tuples, each corresponding to one match of the pattern.
    """
    pattern = None
    if smiles is not None:
        pattern = Chem.MolFromSmiles(smiles)
        pattern = Chem.AddHs(pattern)  # Ensure hydrogens are added for matching
    if smarts is not None:
        if pattern is not None:
            raise ValueError("Can only specify one pattern")
        pattern = Chem.MolFromSmarts(smarts)
    if mol is not None:
        if pattern is not None:
            raise ValueError("Can only specify one pattern")
        pattern = mol
    if fragment is not None:
        if pattern is not None:
            raise ValueError("Can only specify one pattern")
        pattern = ase2rdkit(fragment, **kwargs)
    if pattern is None:
        raise ValueError("Must specify a pattern")

    Chem.SanitizeMol(pattern)

    mol = ase2rdkit(atoms, **kwargs)
    matches = mol.GetSubstructMatches(pattern)
    return matches


def get_substructures(
    atoms: ase.Atoms,
    **kwargs,
) -> list[ase.Atoms]:
    """
    Extract all matched substructures from an ASE Atoms object.

    Parameters
    ----------
    atoms : ase.Atoms
        The structure to search in.
    smarts : str, optional
        A SMARTS string to match substructures.
    smiles : str, optional
        A SMILES string to match substructures.
    mol : Chem.Mol, optional
        An RDKit Mol object to match substructures.
    fragment : ase.Atoms, optional
        A specific ASE Atoms object to match against the structure.
    **kwargs
        Additional keyword arguments passed to `match_substructure`.

    Returns
    -------
    list of ase.Atoms
        List of substructure fragments matching the pattern.
    """
    return [atoms[match] for match in match_substructure(atoms, **kwargs)]


def iter_fragments(atoms: ase.Atoms) -> list[ase.Atoms]:
    """
    Iterate over connected molecular fragments in an ASE Atoms object.

    If a 'connectivity' field is present in `atoms.info`, it will be used
    to determine fragments. Otherwise, `ase.build.separate` will be used.

    Parameters
    ----------
    atoms : ase.Atoms
        A structure that may contain one or more molecular fragments.

    Yields
    ------
    ase.Atoms
        Each connected component (fragment) in the input structure.
    """
    if "connectivity" in atoms.info:
        # connectivity is a list of tuples (i, j, bond_type)
        connectivity = atoms.info["connectivity"]
        for component in find_connected_components(connectivity):
            yield atoms[list(component)]
    else:
        for molecule in separate(atoms):
            yield molecule


def select_atoms_flat_unique(
    mol: Chem.Mol,
    smarts_or_smiles: str,
    hydrogens: tp.Literal["include", "exclude", "isolated"] = "exclude",
) -> list[int]:
    """
    Selects a unique list of atom indices in a molecule using SMARTS or mapped SMILES.
    If the pattern contains atom maps (e.g., [C:1]), only the mapped atoms are returned.
    Otherwise, all atoms in the matched substructure are returned.

    Parameters
    ----------
    mol : Chem.Mol
        RDKit molecule, which can contain explicit hydrogens.
    smarts_or_smiles : str
        SMARTS (e.g., "[F]") or SMILES with atom maps (e.g., "C1[C:1]OC(=[O:1])O1").
    hydrogens : {"include", "exclude", "isolated"}, default "exclude"
        How to handle hydrogens in the final returned list.
        - "include": Include hydrogens attached to matched heavy atoms
        - "exclude": Exclude all hydrogens from results (default)
        - "isolated": Return only hydrogens attached to matched heavy atoms

    Returns
    -------
    list[int]
        A single, flat list of unique integer atom indices matching the criteria.

    Raises
    ------
    ValueError
        If the SMARTS/SMILES pattern is invalid.
    """
    patt = Chem.MolFromSmarts(smarts_or_smiles)

    if not patt:
        raise ValueError(f"Invalid SMARTS/SMILES: {smarts_or_smiles}")

    # Check if the pattern has any mapped atoms.
    mapped_pattern_indices = [
        atom.GetIdx() for atom in patt.GetAtoms() if atom.GetAtomMapNum() > 0
    ]

    matches = mol.GetSubstructMatches(patt)

    # 1. Get the core set of atoms. If the pattern is mapped, only use the
    #    indices corresponding to the mapped atoms. Otherwise, use all atoms.
    core_atom_indices = set()
    if mapped_pattern_indices:
        # Only collect indices from the molecule that correspond to a mapped atom in the pattern.
        for match_tuple in matches:
            for pattern_idx in mapped_pattern_indices:
                core_atom_indices.add(match_tuple[pattern_idx])
    else:
        # Original behavior: get all atoms from all matches if no maps are present.
        core_atom_indices = set(idx for match_tuple in matches for idx in match_tuple)

    # 2. Handle the `hydrogens` parameter based on this core set of atoms.
    if hydrogens == "include":
        final_indices = set(core_atom_indices)
        for idx in core_atom_indices:
            atom = mol.GetAtomWithIdx(idx)
            if atom.GetAtomicNum() != 1:
                for neighbor in atom.GetNeighbors():
                    if neighbor.GetAtomicNum() == 1:
                        final_indices.add(neighbor.GetIdx())
        return sorted(list(final_indices))

    elif hydrogens == "exclude":
        heavy_only = {
            idx
            for idx in core_atom_indices
            if mol.GetAtomWithIdx(idx).GetAtomicNum() != 1
        }
        return sorted(list(heavy_only))

    elif hydrogens == "isolated":
        isolated_hydrogens = set()
        heavy_core_atoms = {
            idx
            for idx in core_atom_indices
            if mol.GetAtomWithIdx(idx).GetAtomicNum() != 1
        }
        for idx in heavy_core_atoms:
            atom = mol.GetAtomWithIdx(idx)
            for neighbor in atom.GetNeighbors():
                if neighbor.GetAtomicNum() == 1:
                    isolated_hydrogens.add(neighbor.GetIdx())
        return sorted(list(isolated_hydrogens))

    return sorted(list(core_atom_indices))


def visualize_selected_molecules(mol: Chem.Mol, a: list[int], b: list[int]):
    """
    Visualizes molecules that contain selected atoms, highlighting the selections.
    Duplicate molecular structures will only be plotted once.

    Parameters
    ----------
    mol : Chem.Mol
        The RDKit molecule object, which may contain multiple fragments.
    a : list[int]
        A list of atom indices to be highlighted in the first color (e.g., pink).
    b : list[int]
        A list of atom indices to be highlighted in the second color (e.g., light blue).

    Returns
    -------
    PIL.Image or None
        A PIL image object of the grid, or None if no molecules to draw.
    """
    # Get separate molecule fragments from the main mol object
    frags = Chem.GetMolFrags(mol, asMols=True)
    frag_indices = Chem.GetMolFrags(mol, asMols=False)

    # --- Step 1: Collect all candidate molecules and their highlight data ---
    candidate_mols = []
    candidate_highlights = []
    candidate_colors = []

    all_selected_indices = set(a + b)

    # Define colors for highlighting
    color_a = (1.0, 0.7, 0.7)  # Pink
    color_b = (0.7, 0.7, 1.0)  # Light Blue

    for i, frag in enumerate(frags):
        original_indices_in_frag = set(frag_indices[i])

        # Check if this fragment contains any of the selected atoms
        if not all_selected_indices.isdisjoint(original_indices_in_frag):
            candidate_mols.append(frag)

            # Map original indices to the new indices within the fragment
            original_to_frag_map = {
                orig_idx: new_idx for new_idx, orig_idx in enumerate(frag_indices[i])
            }

            current_highlights = []
            current_colors = {}

            for idx in a:
                if idx in original_to_frag_map:
                    frag_idx = original_to_frag_map[idx]
                    current_highlights.append(frag_idx)
                    current_colors[frag_idx] = color_a

            for idx in b:
                if idx in original_to_frag_map:
                    frag_idx = original_to_frag_map[idx]
                    if frag_idx not in current_highlights:
                        current_highlights.append(frag_idx)
                    current_colors[frag_idx] = color_b  # Color b takes precedence

            candidate_highlights.append(current_highlights)
            candidate_colors.append(current_colors)

    if not candidate_mols:
        print("No molecules to draw with the given selections.")
        return None

    # --- Step 2: Filter for unique molecules using canonical SMILES ---
    mols_to_draw = []
    highlight_lists = []
    highlight_colors = []
    seen_smiles = set()

    for i, candidate_mol in enumerate(candidate_mols):
        # Generate canonical SMILES to identify unique structures
        mol_no_hs = Chem.RemoveHs(candidate_mol)
        smi = Chem.MolToSmiles(mol_no_hs, canonical=True)

        if smi not in seen_smiles:
            seen_smiles.add(smi)
            mols_to_draw.append(candidate_mol)
            highlight_lists.append(candidate_highlights[i])
            highlight_colors.append(candidate_colors[i])

    # Draw the grid
    img = Draw.MolsToGridImage(
        mols_to_draw,
        molsPerRow=4,
        subImgSize=(200, 200),
        legends=[f"Molecule {i}" for i in range(len(mols_to_draw))],
        highlightAtomLists=highlight_lists,
        highlightAtomColors=highlight_colors,
    )
    return img
