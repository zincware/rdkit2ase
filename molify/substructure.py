import typing as tp

import ase
import matplotlib.pyplot as plt
from ase.build import separate
from rdkit import Chem
from rdkit.Chem import Draw

from molify.ase2x import ase2rdkit
from molify.utils import find_connected_components


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


def select_atoms_grouped(  # noqa: C901
    mol: Chem.Mol,
    smarts_or_smiles: str,
    hydrogens: tp.Literal["include", "exclude", "isolated"] = "exclude",
) -> list[list[int]]:
    """Selects atom indices using SMARTS or SMILES, grouped by disconnected fragments.

    This function identifies all substructure matches and returns a list of atom index
    lists. Each inner list corresponds to a unique, disconnected molecular fragment
    that contained at least one match.

    If the pattern contains atom maps (e.g., "[C:1]", "[C:2]"), only the mapped atoms
    are returned, ordered by their map numbers. Map numbers must be unique within
    the pattern. Otherwise, all atoms in the matched substructures are returned.

    Parameters
    ----------
    mol : rdchem.Mol
        RDKit molecule, which can contain multiple disconnected fragments and
        explicit hydrogens.
    smarts_or_smiles : str
        SMARTS pattern (e.g., "[F]") or SMILES with atom maps
        (e.g., "CC(=O)N[C:1]([C:2])[C:3](=O)[N:4]C"). When using mapped atoms,
        map numbers must be unique.
    hydrogens : {'include', 'exclude', 'isolated'}, default='exclude'
        How to handle hydrogens in the final returned list for each group:
            - 'include': Add hydrogens bonded to selected heavy atoms after each
            mapped atom.
            - 'exclude': Remove all hydrogens from the selection.
            - 'isolated': Returns only the hydrogens that are bonded
            to selected heavy atoms.


    Returns
    -------
    list[list[int]]
        A list of integer lists. Each inner list contains the atom indices
        for a matched, disconnected fragment. For mapped patterns, atoms are ordered
        by their map numbers. Fragments with no matches are omitted from the output.

    Raises
    ------
    ValueError
        If the provided SMARTS/SMILES pattern is invalid or if atom map labels
        are used multiple times within the same pattern.

    Examples
    --------
    >>> # Molecule with two disconnected fragments: ethanol and fluoromethane
    >>> mol = Chem.MolFromSmiles("CCO.CF")  # Indices: C(0)C(1)O(2) . C(3)F(4)
    >>>
    >>> # Select all carbon atoms
    >>> select_atoms_grouped(mol, "[C]")
    [[0, 1], [3]]
    >>>
    >>> # Select fluorine and its bonded carbon using 'include'
    >>> select_atoms_grouped(mol, "[F]", hydrogens="include")
    [[3, 4]]

    """
    patt = Chem.MolFromSmarts(smarts_or_smiles)
    if patt is None:
        # Support mapped SMILES patterns too
        patt = Chem.MolFromSmiles(smarts_or_smiles)
    if patt is None:
        raise ValueError(f"Invalid SMARTS/SMILES: {smarts_or_smiles}")

    # Get mapped indices from the pattern, if any, and validate uniqueness
    mapped_pattern_indices = []
    atom_map_numbers = []
    for atom in patt.GetAtoms():
        if atom.GetAtomMapNum() > 0:
            map_num = atom.GetAtomMapNum()
            if map_num in atom_map_numbers:
                raise ValueError(f"Label '{map_num}' is used multiple times")
            atom_map_numbers.append(map_num)
            mapped_pattern_indices.append(atom.GetIdx())

    # If we have mapped atoms, we need to sort them by their
    #  map numbers to preserve order
    if mapped_pattern_indices:
        # Create pairs of (map_number, pattern_index) and sort by map_number
        map_index_pairs = [
            (patt.GetAtomWithIdx(idx).GetAtomMapNum(), idx)
            for idx in mapped_pattern_indices
        ]
        map_index_pairs.sort(key=lambda x: x[0])  # Sort by map number
        mapped_pattern_indices = [idx for _, idx in map_index_pairs]

    # Find all matches in the entire molecule just once for efficiency
    all_matches = mol.GetSubstructMatches(patt)
    if not all_matches:
        return []

    # Get the indices of atoms in each disconnected fragment
    fragment_sets = [set(frag) for frag in Chem.GetMolFrags(mol, asMols=False)]

    grouped_indices = []
    for fragment_atom_indices in fragment_sets:
        # Filter matches to include only those fully contained within the fragment
        fragment_matches = [
            match for match in all_matches if set(match).issubset(fragment_atom_indices)
        ]

        if not fragment_matches:
            continue

        # 1. Get the core set of atoms for this fragment. If the pattern is mapped,
        #    use only the indices corresponding to mapped atoms. Otherwise, use all.
        if mapped_pattern_indices:
            # For mapped patterns, preserve the order of atoms based
            # on their map numbers
            core_atom_indices_ordered = []
            for match_tuple in fragment_matches:
                match_atoms = [
                    match_tuple[pattern_idx] for pattern_idx in mapped_pattern_indices
                ]
                core_atom_indices_ordered.extend(match_atoms)
            # Remove duplicates while preserving order
            seen = set()
            core_atom_indices_ordered = [
                x for x in core_atom_indices_ordered if not (x in seen or seen.add(x))
            ]
            core_atom_indices = set(core_atom_indices_ordered)
        else:
            core_atom_indices = {
                idx for match_tuple in fragment_matches for idx in match_tuple
            }
            core_atom_indices_ordered = sorted(core_atom_indices)

        if not core_atom_indices:
            continue

        # 2. Handle the `hydrogens` parameter for this fragment's core atoms
        if hydrogens not in ("include", "exclude", "isolated"):
            raise ValueError(
                f"Invalid value for `hydrogens`: {hydrogens!r}. "
                "Expected one of 'include', 'exclude', 'isolated'."
            )

        if hydrogens == "include":
            # Include both core atoms and their hydrogens, maintaining order
            final_indices_ordered = []
            for idx in core_atom_indices_ordered:
                # Add the core atom first
                final_indices_ordered.append(idx)
                # Then add its hydrogens
                atom = mol.GetAtomWithIdx(idx)
                if atom.GetAtomicNum() != 1:  # is a heavy atom
                    hydrogen_indices = [
                        neighbor.GetIdx()
                        for neighbor in atom.GetNeighbors()
                        if neighbor.GetAtomicNum() == 1
                    ]
                    final_indices_ordered.extend(sorted(hydrogen_indices))

        elif hydrogens == "exclude":
            # Only heavy atoms from core selection
            final_indices_ordered = [
                idx
                for idx in core_atom_indices_ordered
                if mol.GetAtomWithIdx(idx).GetAtomicNum() != 1
            ]

        elif hydrogens == "isolated":
            # Only hydrogens bonded to core heavy atoms, maintaining order
            final_indices_ordered = []
            for idx in core_atom_indices_ordered:
                atom = mol.GetAtomWithIdx(idx)
                if atom.GetAtomicNum() != 1:  # is a heavy atom
                    hydrogen_indices = [
                        neighbor.GetIdx()
                        for neighbor in atom.GetNeighbors()
                        if neighbor.GetAtomicNum() == 1
                    ]
                    final_indices_ordered.extend(sorted(hydrogen_indices))

        # Only add the group if it contains any atoms after processing
        if final_indices_ordered:
            grouped_indices.append(final_indices_ordered)

    return grouped_indices


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
    grouped_indices = select_atoms_grouped(mol, smarts_or_smiles, hydrogens=hydrogens)
    if not grouped_indices:
        return []

    # Flatten the list of lists and remove duplicates
    unique_indices = set()
    for group in grouped_indices:
        unique_indices.update(group)

    return sorted(unique_indices)


def _collect_highlighted_fragments(mol, args, alpha):
    """Helper function to collect and process fragment highlights."""
    frags = Chem.GetMolFrags(mol, asMols=True)
    frag_indices = Chem.GetMolFrags(mol, asMols=False)

    candidate_mols = []
    candidate_highlights = []
    candidate_colors = []

    # Collect all selected indices from all argument lists
    all_selected_indices = set()
    for atom_list in args:
        all_selected_indices.update(atom_list)

    # Get colors from matplotlib's tab10 colormap and add alpha
    colors = plt.cm.tab10.colors
    highlight_colors = [colors[i % len(colors)] + (alpha,) for i in range(len(args))]

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

            # Process each argument list with its corresponding color
            for arg_idx, atom_list in enumerate(args):
                color = highlight_colors[arg_idx]
                for idx in atom_list:
                    if idx in original_to_frag_map:
                        frag_idx = original_to_frag_map[idx]
                        if frag_idx not in current_highlights:
                            current_highlights.append(frag_idx)
                        # Later argument lists take precedence for coloring
                        current_colors[frag_idx] = color

            candidate_highlights.append(current_highlights)
            candidate_colors.append(current_colors)

    return candidate_mols, candidate_highlights, candidate_colors


def _filter_unique_molecules(candidate_mols, candidate_highlights, candidate_colors):
    """Helper function to filter for unique molecular structures."""
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

    return mols_to_draw, highlight_lists, highlight_colors


def visualize_selected_molecules(
    mol: Chem.Mol,
    *args,
    mols_per_row: int = 4,
    sub_img_size: tuple[int, int] = (200, 200),
    legends: list[str] | None = None,
    alpha: float = 0.5,
):
    """
    Visualizes molecules with optional atom highlighting.
    If no atom selections are provided, displays the molecule without highlights.
    Duplicate molecular structures will only be plotted once.

    Parameters
    ----------
    mol : Chem.Mol
        The RDKit molecule object, which may contain multiple fragments.
    *args : list[int]
        Variable number of lists containing atom indices to be highlighted.
        Each list will be assigned a different color from matplotlib's tab10 colormap.
        If no arguments provided, displays the molecule without highlights.
    mols_per_row : int, default 4
        Number of molecules per row in the grid.
    sub_img_size : tuple[int, int], default (200, 200)
        Size of each molecule image.
    legends : list[str] | None, default None
        Custom legends for each molecule. If None, default legends will be used.
    alpha : float, default 0.5
        Transparency level for the highlighted atoms (0.0 = fully transparent,
        1.0 = opaque).

    Returns
    -------
    PIL.Image
        A PIL image object of the grid.
    """
    # Handle empty args case - display molecule without highlights
    if not args:
        img = Draw.MolsToGridImage(
            [mol],
            molsPerRow=mols_per_row,
            subImgSize=sub_img_size,
            legends=legends if legends is not None else ["Molecule 0"],
        )
        return img

    # Collect highlighted fragments
    candidate_mols, candidate_highlights, candidate_colors = (
        _collect_highlighted_fragments(mol, args, alpha)
    )

    if not candidate_mols:
        print("No molecules to draw with the given selections.")
        return None

    # Filter for unique molecules
    mols_to_draw, highlight_lists, highlight_colors = _filter_unique_molecules(
        candidate_mols, candidate_highlights, candidate_colors
    )

    # Draw the grid
    final_legends = (
        legends
        if legends is not None
        else [f"Molecule {i}" for i in range(len(mols_to_draw))]
    )

    img = Draw.MolsToGridImage(
        mols_to_draw,
        molsPerRow=mols_per_row,
        subImgSize=sub_img_size,
        legends=final_legends,
        highlightAtomLists=highlight_lists,
        highlightAtomColors=highlight_colors,
    )
    return img
