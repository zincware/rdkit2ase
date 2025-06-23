______________________________________________________________________

title: 'rdkit2ase: Molecular Structure Generation and Manipulation for
Machine-Learned Interatomic Potentials' tags:

- Python
- cheminformatics
- MLIPs
- ASE
- RDKit
- PACKMOL
- NetworkX authors:
- name: Fabian Zills orcid: 0000-0002-6936-4692 affiliation: "1" affiliations:
- name: Institute for Computational Physics, University of Stuttgart, 70569
  Stuttgart, Germany index: 1 date: 2025-06-23 bibliography: bibliography.bib

______________________________________________________________________

# Summary

The increasing prevalence of Machine-Learned Interatomic Potentials (MLIPs) has
shifted requirements for setting up atomistic simulations. Unlike classical
force fields, MLIPs primarily require atomic positions and species, thereby
removing the need for predefined topology files used for classical force fields
in molecular dynamics software like GROMACS[@abrahamGROMACSHighPerformance2015],
LAMMPS[@LAMMPS], ESPResSo[@weikESPResSo40Extensible2019] or OpenMM
[@eastmanOpenMM8Molecular2024]. Consequently, the Atomic Simulation Environment
(ASE) [@larsenAtomicSimulationEnvironment2017] has become a popular Python
toolkit for handling atomic structures and interfacing with MLIPs, particularly
within the materials science and soft matter communities, because it originates
from _ab initio_ simulations, which share the same setup as MLIP-driven studies.

Concurrently, RDKit [@landrumRdkitRdkit2023_03_22023] offers extensive
functionality for cheminformatics and manipulating chemical structures. However,
standard RDKit workflows are not designed for MLIP-driven simulation, while
typical ASE-MLIP workflows may lack rich, explicit chemical information such as
bond orders or molecular identities, as well as generating different
conformations or searching substructures.

The rdkit2ase package bridges this gap, providing an interface between RDKit's
chemical structure generation and cheminformatics capabilities and ASE's
handling of 3D atomic structures. Furthermore, rdkit2ase integrates with PACKMOL
[@martinezPACKMOLPackageBuilding2009] to facilitate the creation of complex,
periodic simulation cells with diverse chemical compositions, all while
preserving crucial chemical connectivity information. Lastly, the combination of
these packages enables selection and manipulation of atomistic structures based
on chemical knowledge rather than manual index handling.

# Statement of need

rdkit2ase serves as a vital link between RDKit, ASE, and PACKMOL. While its core
function is to interface these tools, it thereby unlocks new capabilities and
significantly reduces the manual coding and data wrangling typically required
for preparing and analyzing molecular simulations.

The package simplifies workflows that previously involved laborious tasks such
as sourcing individual structure files from various databases (e.g., the
Materials Project [@jainCommentaryMaterialsProject2013] or the ZINC database
[@tingleZINC22AFreeMultiBillionScale2023]) and custom setups of simulation
cells. This simplification not only accelerates research but also supports the
setup of more complex and chemically diverse simulation scenarios.

One challenge in MLIP-driven simulations is the post-simulation identification
and analysis of molecular fragments or chemical changes, as explicit topological
information is often absent. rdkit2ase addresses this by enabling the use of
RDKit's powerful SMILES[@weiningerSMILESChemicalLanguage1988]/SMARTS-based
substructure searching on ASE structures. In addition, the resulting molecular
graph can be exported to a NetworkX[@hagbergExploringNetworkStructure2008]
object for further analysis. This selection and handling allows for similar
functionality as is provided by the
MDAnalysis[@gowersMDAnalysisPythonPackage2016] atom selection language, targeted
towards simulations with a fixed topology.

# Features and Implementation

![Visualization of a 3D structure from ASE, visualized with ZnDraw [@elijosiusZeroShotMolecular2024] (left) and its corresponding RDKit 2D chemical structure representation (right).abel{fig:zndraw-rdkit}](zndraw_rdkit.svg)

The generation of atomic configurations in rdkit2ase is centered around SMILES
for defining molecular species. A typical workflow involves:

1. Generating 3D conformers for individual molecular species from their SMILES
   using RDKit.
1. Packing these conformers into a simulation box to achieve a target density
   using PACKMOL and obtaining an ASE Atoms object representing the simulation
   cell, ready for use with MLIPs.
1. Post-processing the simulation data by identifying and selecting structures
   based on SMARTS.

```python
from rdkit2ase import pack, smiles2conformers

water = smiles2conformers("O", numConfs=2)
print(water[0].info['connectivity'])
>>> [(0, 1, 1.0), (0, 2, 1.0)] # (atom_idx1, atom_idx2, bond_order)
ethanol = smiles2conformers("CCO", numConfs=5)
density = 1000  # kg/m^3
box = pack([water, ethanol], [7, 5], density, packmol="packmol.jl")
print(box)
>>> Atoms(symbols='C10H44O12', pbc=True, cell=[8.4, 8.4, 8.4])
```

All ASE Atoms objects generated or processed by rdkit2ase will store
`connectivity` information (bonds and their orders) within the `ase.Atoms.info`
dictionary. If available, rdkit2ase uses this bond information for accurate
interconversion. If an ASE structure is converted to an RDKit molecule without
pre-existing connectivity, rdkit2ase leverages RDKit's robust bond perception
algorithms [@kimUniversalStructureConversion2015] to estimate this information.
A representation from both packages is shown in \\autoref{fig:zndraw-rdkit}.

```python
from rdkit2ase import ase2rdkit
from rdkit.Chem import Draw

mol = ase2rdkit(box)
img = Draw.MolToImage(mol)
```

This bidirectional conversion capability allows the use of RDKit's chemical
analysis tools together with ASE for MLIP-based simulations..

For instance, if during a simulation, atomic positions in an ASE Atoms object
are updated, rdkit2ase can convert this structure back to an RDKit molecule to
analyze chemical changes or identify specific substructures. One common example
is the extraction of substructures based on SMILES or SMARTS to track their
structure and dynamics within a simulation. For example, rdkit2ase streamlines
the extraction of the CH$\_3$ alkyl group from the ethanol molecules inside the
simulation cell, without manual index lookup.

```py
from rdkit2ase import get_substructures

frames: list[ase.Atoms] = get_substructures(
  atoms=box,
  smiles="[C]([H])([H])[H]"
)
```

# Acknowledgements

F. Z. acknowledges support by the Deutsche Forschungsgemeinschaft (DFG, German
Research Foundation) in the framework of the priority program SPP 2363,
“Utilization and Development of Machine Learning for Molecular Applications –
Molecular Machine Learning” Project No. 497249646. Further funding through the
DFG under Germany's Excellence Strategy – EXC 2075 – 390740016 and the Stuttgart
Center for Simulation Science (SimTech) was provided.

# Related software

The functionality of rdkit2ase relies critically on the following packages:

- [RDKit](https://www.rdkit.org/docs/index.html): For cheminformatics tasks,
  SMILES parsing, conformer generation, and substructure searching.
- [ASE](https://wiki.fysik.dtu.dk/ase/): For representing and manipulating
  atomic structures, and interfacing with simulation engines.
- [PACKMOL](https://m3g.github.io/packmol/): For packing molecules into
  simulation boxes. rdkit2ase can interface with either a PACKMOL executable or
  the packmol.jl package.
- [NetworkX](https://networkx.org/): For the handling and analysis of molecular
  graphs.

The rdkit2ase package is currently a crucial part of the following software
packages:

- [IPSuite](https://github.com/zincware/ipsuite): For generating structures for
  training MLIPs.
- [ZnDraw](https://github.com/zincware/zndraw): Interactive generation of
  simulation boxes and selection of substructures through a graphical user
  interface inside a web-based visualization package.
- [mlipx](https://github.com/basf/mlipx): Creating initial structures for
  benchmarking different MLIPs on real-world test scenarios.
