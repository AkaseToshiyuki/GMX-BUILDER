# Local CHARMM coverage — GMXBUILDER 1.0.0

Automatic identification can recognize chemistry beyond this parameter catalog.
It does not imply that a corresponding local CHARMM model is available.
Coverage means the exact molecular state below, not every molecule with the same functional group.
Use the research checkbox only within the separately documented [research domains](README.md).
Both releases were exercised with real GROMACS preprocessing and static topology checks.
Atom-mapped SMILES may be required to distinguish protonation/tautomer sites.
| Identity | RTP | Explicit state (SMILES) | CHARMM36m | CHARMM36 |
|---|---|---|---|---|
| methanol | MEOH | `CO` | Supported | Supported |
| ethanol | ETOH | `CCO` | Supported | Supported |
| benzene | BENZ | `c1ccccc1` | Supported | Supported |
| methylamine | MAM1 | `CN` | Supported | Supported |
| methylammonium | MAMM | `C[NH3+]` | Supported | Supported |
| ethylammonium | EAMM | `CC[NH3+]` | Supported | Supported |
| dimethylamine | DMAM | `CNC` | Supported | Supported |
| dimethylammonium | MMAM | `C[NH2+]C` | Supported | Supported |
| acetamide | ACEM | `CC(N)=O` | Supported | Blocked: inequivalent H / absent HDB |
| propane | PRPA | `CCC` | Supported | Supported |
| dimethyl ether | DMEE | `COC` | Supported | Supported |
| diethyl ether | DETE | `CCOCC` | Supported | Supported |
| 1-propanol | PROH | `CCCO` | Supported | Supported |
| ethyl methyl ether | METE | `CCOC` | Supported | Supported |
| ethylbenzene | EBEN | `CCc1ccccc1` | Supported | Supported |
| toluene | TOLU | `Cc1ccccc1` | Supported | Supported |
| cumene | CUME | `CC(C)c1ccccc1` | Supported | Supported |
| benzylammonium | BZAM | `[NH3+]Cc1ccccc1` | Supported | Supported |
| acetone | ACO | `CC(=O)C` | Supported | Supported |
| acetaldehyde | AALD | `CC=O` | Supported | Supported |
| acetonitrile | ACN | `CC#N` | Supported | Blocked: missing native torsion |
| N-methylacetamide | NMA | `CNC(C)=O` | Supported | Supported |
| N,N-dimethylacetamide | DMA | `CC(=O)N(C)C` | Supported | Supported |
| N,N-dimethylformamide | DMF | `CN(C)C=O` | Supported | Supported |
| ethyl acetate | ETAC | `CCOC(C)=O` | Supported | Supported |
| methyl acetate | MAS | `COC(C)=O` | Supported | Supported |
| pyridine | PYR1 | `n1ccccc1` | Supported | Supported |
| pyrimidine | PYRM | `n1cnccc1` | Supported | Supported |
| furan | FURA | `o1cccc1` | Supported | Supported |
| thiophene | THIP | `s1cccc1` | Supported | Supported |
| thiazole | THAZ | `s1cncc1` | Supported | Supported |
| isothiazole | ISOT | `s1nccc1` | Supported | Supported |
| indole | INDO | `c1ccc2[nH]ccc2c1` | Supported | Supported |
| benzofuran | ZFUR | `c1ccc2occc2c1` | Supported | Supported |
| benzothiophene | ZTHP | `c1ccc2sccc2c1` | Supported | Supported |
| benzothiazole | ZTHZ | `c1ccc2scnc2c1` | Supported | Supported |
| phenol | PHEN | `Oc1ccccc1` | Supported | Supported |
| fluorobenzene | FLUB | `Fc1ccccc1` | Supported | Supported |
| methanethiol | MESH | `CS` | Supported | Supported |
| ethanethiol | ETSH | `CCS` | Supported | Supported |
| dimethyl sulfone | DMSN | `CS(C)(=O)=O` | Supported | Supported |
| dimethyl phosphate anion | DMEP | `COP(=O)([O-])OC` | Supported | Supported |
| methyl phosphate | MP_0 | `COP(=O)(O)O` | Supported | Supported |
| methyl phosphate monoanion | MP_1 | `COP(=O)(O)[O-]` | Blocked: phosphate resonance/stereo normalization | Blocked: phosphate resonance/stereo normalization |
| methyl phosphate dianion | MP_2 | `COP(=O)([O-])[O-]` | Supported | Supported |
| imidazole | IMIA | `c1ncc[nH]1` | Supported | Supported |
| imidazolium | IMIM | `c1[nH+]cc[nH]1` | Supported | Supported |
| morpholinium | MORP | `O1CC[NH2+]CC1` | Supported | Supported |
| piperidinium | PIP | `[NH2+]1CCCCC1` | Supported | Supported |
| pyrrolidine | PRLD | `N1CCCC1` | Supported | Supported |
| pyrrolidinium | PRLP | `[NH2+]1CCCC1` | Supported | Supported |
| benzoate | 3CB | `O=C([O-])c1ccccc1` | Supported | Supported |
| benzoic acid | ZOIC | `O=C(O)c1ccccc1` | Supported | Supported |
| acetic acid | ACEH | `CC(=O)O` | Supported | Supported |
| acetate | ACET | `CC(=O)[O-]` | Supported | Supported |
| formamide | FORM | `NC=O` | Supported | Supported |
| urea | UREA | `NC(N)=O` | Supported | Supported |
| N-benzylacetamide | NZAD | `CC(=O)NCc1ccccc1` | Supported | Supported |
| phenylacetic acid | BZAA | `O=C(O)Cc1ccccc1` | Supported | Supported |
| phenylacetate | BZAC | `O=C([O-])Cc1ccccc1` | Supported | Supported |
| N,N-prime-dimethylurea | 12MU | `CNC(=O)NC` | Supported | Supported |

“Supported” establishes complete topology construction, not independent physical validation.
Native HDB is used where reliable. Legacy ligands and the Jul2022 ETAC/ETSH
HDB naming defects use RDKit hydrogen coordinates only when hydrogens on each
parent have equal types/charges. Explicit RTP atom names bypass legacy ARN
renaming (needed for NMA). The model never repairs a missing complete-template
parameter with research analogy.

New-scaffold research examples include S/R amphetamine (+1), propylammonium,
propylbenzene and 3-phenylpropylammonium, in addition to the earlier linear
C/O examples. They have complete construction checks, but no physical QA.
A new heterocyclic/amide/sulfur/phosphate scaffold does **not** inherit coverage
from this catalog: it currently requires a matching full template or CGenFF import.
