# GMXBUILDER Scientific Compatibility and Limitations

<p><strong>English</strong> · <a href="SCIENTIFIC_COMPATIBILITY.zh-CN.md">简体中文</a></p>

This document explains which combinations GMXBUILDER can build and what a
successful build does not prove. The runtime capability registry in the
installed version is authoritative; fixed counts of lipids or modifications
are intentionally not duplicated here.

## 1. Query installed capabilities

```bash
gmxbuilder list-ff
gmxbuilder list-water
gmxbuilder list-lipids
gmxbuilder lipid-library status
```

The corresponding discovery endpoints are:

```text
GET /api/options
GET /api/patches?force_field=<name>
GET /api/crosslink-capabilities?force_field=<name>
GET /api/terminal-capabilities?force_field=<name>
GET /api/lipid-library-status?lipid_name=<name>&force_field=<name>&lipid_ff=<backend>
GET /api/coarse-grained/capabilities
```

An item appearing in the interface does not mean that it is compatible with
the currently selected force field. Disabled items and server errors describe
the available alternatives.

## 2. Force-field families

| Protein family | Membrane | Retained small molecules | Contract |
|---|---|---|---|
| CHARMM36m / CHARMM36 | Validated CHARMM36 lipid parameters | Exact CHARMM templates or matching user-supplied CGenFF MOL2+STR files | Molecular identity, net charge, and penalty are checked |
| Amber14SB / Amber99SB / Amber99SB-ILDN | One coherent Lipid21 or validated GAFF2 membrane | GAFF2 + AM1-BCC | Each GAFF2 molecule requires an explicit integer net charge |
| OPLS-AA | Exact installed OPLS lipid parameters only | Exact installed OPLS parameters only | No general membrane fallback is currently installed |

CHARMM proteins cannot be combined with GAFF membranes or ligands, and Amber
proteins cannot be combined with CHARMM/CGenFF membranes or ligands. Any
combination that would introduce conflicting GROMACS `[ defaults ]` is rejected.
Successful file inclusion alone does not establish cross-family compatibility.

The water model is locked with the complete force-field combination during the
Force Field step and cannot be silently replaced during solvation.

## 3. Lipids and the pre-equilibrated library

Amber membranes resolve in this order: exact Lipid21, one coherent GAFF2
membrane, or unavailable. A mixed membrane never combines Lipid21 and GAFF2
per molecule. CHARMM36 and CHARMM36m use separately identified strict
libraries even where individual lipid parameters are similar.

A strict library entry is usable only when all of the following are true:

- force-field family, schema, canonical molecular identity, and topology/atom
  order signatures match;
- the structure comes from an explicit-solvent, semi-isotropic NPT workflow;
- conformer counts and metadata are complete; and
- APL, DHH, orientation, and hydrophobic-core quality gates pass.

Geometry-only bootstrap conformers are not shipped or advertised as an
equilibrated library. A topology whose conformer fails the quality gates remains unavailable;
GMXBUILDER does not substitute an approximate chain length or similarly named
molecule.

The release archive is a validated subset, not a promise that every compatible
registry entry passed equilibration. Its checksum, strict-library schema, and
each included library entry are verified before installation. Combinations
that failed or have not completed the production quality gates are excluded
and remain unavailable in the interface.

Verify and install release assets with:

```bash
gmxbuilder prebuilt-assets status
gmxbuilder prebuilt-assets install
gmxbuilder lipid-library status
```

Administrators can maintain the global library with
`gmxbuilder lipid-library build/queue/status`. Short `--test-mode` outputs are
smoke-test material and do not pass the production runtime gate.

### Task-private custom lipids

Custom lipids currently require Amber + GAFF2. Canonical stereochemical
identity and InChIKey checks prevent re-submission of an installed molecule.
The task remains blocked until GAFF2/AM1-BCC parameterization and explicit-
solvent NPT pre-equilibration pass. Parameters and conformers remain private to
that Task and are removed when the Task expires.

## 4. Small molecules

- GAFF2 requires the user to confirm the integer net charge of every retained
  molecule. Automated suggestions do not replace judgement about pH,
  tautomers, salt form, or coordination state.
- GAFF2 may add hydrogens, but parameterization must preserve the input heavy-
  atom identity and order.
- CHARMM small molecules require CGenFF MOL2 and STR output for the same
  chemical structure. High-penalty parameters require external quantum-
  chemical validation or refitting.
- Metal coordination, covalent ligands, reactive intermediates, and coupled
  protonation are not general automated parameterization capabilities.

## 5. Nucleic acids

Canonical linear DNA and RNA are supported only by the Solvator workflow with
CHARMM36m and CHARMM TIP3P. GMXBUILDER treats each chain as a polymer and uses
the bundled GROMACS/CHARMM36 databases to construct 5′/3′ hydroxyl termini,
hydrogens, O3′–P links, bonded terms, and an integral chain charge. Protein–DNA,
protein–RNA, and compatible non-covalent CHARMM ligand complexes are supported.
This native preparation replaces the uploaded nucleic-acid coordinates with
the hydrogen-complete `pdb2gmx` coordinates; the Step 3 viewer is the required
coordinate review point.

Backbone discontinuities, circular chains, covalent DNA/RNA hybrids, and
modified or noncanonical nucleotides are rejected. Amber nucleic-acid models,
membrane-embedded nucleic acids, and Martini nucleic acids are not currently
available. Nucleotide-like free ligands remain in the small-molecule workflow
instead of being silently attached to a polymer.

## 6. Protein protonation, termini, and modifications

PROPKA output is a discrete suggestion for one static structure, not constant-
pH MD. It does not jointly solve membrane potential, ligand protonation, metal
coordination, or multiple conformations. Catalytic sites, buried hydrogen-bond
networks, and cofactors require manual review.

Standard NH3+/COO− termini are supported. ACE/NME caps are inserted only when
the selected force field contains an atom-complete template. Residue
modifications are enabled from native templates for the selected force field,
not reused across force-field families.

Representative supported capabilities include Amber Ser/Thr/Tyr
phosphorylation states; CHARMM36m phosphorylation and selected Lys/Arg/Cys,
Tyr, and Ser modifications; explicit stereochemistry for R-methionine
sulfoxide, trans-(2S,4R)-hydroxyproline, and hydroxylysine; deamidation in all
bundled protein force fields; and validated Amber CYS→CYX disulfide pairs.

Every enabled modification must have a unique chemical identity, charge, and
stereochemistry; complete atom and local-geometry operations; complete RTP,
HDB, bonded, and non-bonded parameters; stable checkpoint identity; and a real
target-force-field `gmx grompp` check. Unsupported glycosylation, long-chain
lipidation, ambiguous approximate templates, and CHARMM disulfide patches stay
explicitly unavailable.

## 7. Martini 3 coarse-grained boundary

Martini 3 is an independent resolution and parameter system. It is not mixed
with the Amber, CHARMM, or OPLS atomistic systems above. The current workflow
uses pinned Martini 3.0.0 assets, Martinize2/Vermouth 0.15.0, and COBY 1.0.14.
It exposes separate Martini 3 Solvent and Bilayer builders. The solvent builder
supports standard proteins in water. The bilayer builder supports flat pure,
mixed, symmetric/asymmetric membranes with an exact requested integer count per
leaflet, plus optional standard proteins. A dedicated orientation step uses the
same PPM-like energy/segment review model as the atomistic membrane workflow,
while allowing an exact manual transform. The periodic box is derived from the
confirmed molecular envelope, padding, and requested membrane size; users do
not enter a box Z that can truncate the positioned protein. Both builders use
regular W water and NA/CL ions.

Ligands, PTMs, glycans, nucleic acids, arbitrary custom CG molecules,
mixed-resolution models, complex curved surfaces, Gō/OLIVES, and backmapping
are unavailable and are rejected during input review. Query
`GET /api/coarse-grained/capabilities` for the authoritative installed list and
see the [User Manual](GMXBUILDER_USER_MANUAL_V1.0.4.md) for operation.

## 8. Build quality and responsibility

A successful GMXBUILDER build means that the input, coordinate checkpoints,
topology, box, index, and MDP files passed the current automated checks and can
enter minimization and equilibration. It does not establish that every mixture,
temperature, phase, pH, conformation, or modification is experimentally
correct, nor that production sampling has converged.

Before production, review total charge, membrane APL/thickness/leaflet
orientation and voids, protein orientation, solvent layers, ion positions, and
ligand charge/parameter penalties. Run minimization and staged equilibration,
and use independent repeats and experimental or literature comparison where
the scientific question requires them.

## 9. References

- [GROMACS force-field overview](https://manual.gromacs.org/documentation/current/user-guide/force-fields.html)
- [GROMACS topology format and defaults](https://manual.gromacs.org/documentation/current/reference-manual/topologies/topology-file-formats.html)
- [Lipid21 validation](https://pubmed.ncbi.nlm.nih.gov/34286854/)
- [GAFF](https://pubmed.ncbi.nlm.nih.gov/15116359/)
- [CGenFF](https://pmc.ncbi.nlm.nih.gov/articles/PMC2888302/)
- [CHARMM36 lipid validation](https://pmc.ncbi.nlm.nih.gov/articles/PMC2922408/)
- [GROMACS pdb2gmx input databases](https://manual.gromacs.org/documentation/current/reference-manual/topologies/pdb2gmx-input-files.html)
- [RCSB Chemical Component Dictionary](https://www.rcsb.org/ligand)
- [Martini 3](https://doi.org/10.1038/s41592-021-01098-3)
- [Martini 3 tutorials](https://cgmartini.nl/docs/tutorials/Martini3/tutorials.html)
