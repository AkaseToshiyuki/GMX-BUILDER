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

Library discovery returns bounded availability summaries (`metadata_scope` is
`availability_summary`), including the validation scope. It does not return full
trajectory evidence or reload coordinate ensembles on each poll. A `checking`
state is not scientific rejection; production construction independently rechecks
current source evidence. Full scientific metadata remains in the library assets.

An item appearing in the interface does not mean that it is compatible with
the currently selected force field. Disabled items and server errors describe
the available alternatives.

## 2. Force-field families

| Protein family | Membrane | Retained small molecules | Contract |
|---|---|---|---|
| CHARMM36m / CHARMM36 | Validated CHARMM36 lipid parameters | Local CHARMM templates / documented research domains, or matching CGenFF MOL2+STR files | Molecular identity, net charge, and penalty are checked |
| Amber14SB / Amber99SB / Amber99SB-ILDN | Lipid21 per covered species; GAFF2 for missing species after mixed-model validation | GAFF2 + AM1-BCC | Each GAFF2 molecule requires an explicit integer net charge |
| OPLS-AA | Exact installed OPLS lipid parameters only | Exact installed OPLS parameters only | No general membrane fallback is currently installed |

CHARMM proteins cannot be combined with GAFF membranes or ligands, and Amber
proteins cannot be combined with CHARMM/CGenFF membranes or ligands. Any
combination that would introduce conflicting GROMACS `[ defaults ]` is rejected.
Successful file inclusion alone does not establish cross-family compatibility.

The water model is locked with the complete force-field combination during the
Force Field step and cannot be silently replaced during solvation. Water-policy
version 1.0 distinguishes `recommended`, regression-`supported`,
`expert-unvalidated`, and `prohibited` combinations. A bundled `.itp` file is
only a technical prerequisite. Expert-unvalidated combinations require the
explicit API/CLI acknowledgement `allow_unvalidated_water_model=true`; that
acknowledgement does not promote the combination to validated status and is
recorded with the task metadata.

## 3. Lipids and validation scopes

Amber membranes prefer exact Lipid21 for each covered species. A mixed
Lipid21/GAFF2 assignment preserves that choice and uses GAFF2 only for missing
species. A whole-GAFF2 model remains an explicit alternative. User mixed-model
initialization requires current validated GAFF2 guest conformers sampled in a
90:10 Lipid21 POPC host; the offline V4 workflow can generate that evidence.
Parameter composition and preprocessing alone do not establish physical validity
of a new mixture. CHARMM36 and CHARMM36m retain separate library identities.

Lipid21 carries explicit per-pair LJ coefficients, charges and electrostatic
scales (GROMACS pair function 2), independent of the parent defaults. The
plasmalogen CHARMM extension applies only to the molecule's vinyl-ether motif;
its shared type names must never override ordinary lipid torsions globally.

Schema 4 requires a content fingerprint of parameter sources, relevant exporter
code and the sampling host. Missing or changed fingerprints invalidate historical
conformations and forbid continuation under new parameters. Release 1.0.0 distributes the V4 lipid library: 37 accepted initialization libraries and
134,000 conformers. See the [release support matrix](release-support.md) for
exact combinations and evidence limits. Incompatible historical assets remain
unavailable.

A strict library entry is usable only when all of the following are true:

- force-field family, schema, canonical molecular identity, and topology/atom
  order signatures and parameter fingerprints match;
- the structure comes from an explicit-solvent, semi-isotropic NPT workflow;
- conformer counts and metadata are complete; and
- APL, DHH, orientation, and hydrophobic-core quality gates pass.

Geometry-only bootstrap conformers are not shipped or advertised as an
equilibrated library. A topology whose conformer fails the quality gates remains unavailable;
GMXBUILDER does not substitute an approximate chain length or similarly named
molecule.

The interface reports a validation scope for each available lipid/parameter
combination. **Initialization conformers** have passed construction admission;
this does not certify bulk membrane equilibrium or area convergence.
**Pre-equilibrated conformers** additionally require the recorded replica
sampling, stationarity, platform, replica-agreement and quantitative-area gates.
Neither scope replaces equilibration of a newly assembled system. A saved
`ready` or `area_converged` flag alone cannot grant the stronger label.

The eight PG species DAPG, DLIPG, DMPG, DOPG, DPPG, PAPG, POPG and SOPG use the
natural R,S model. Available Amber Lipid21 PG templates are regenerated from the
official **PGS** head module, including its coordinates and coefficients; the PGR
coordinate seed is not the selected isomer. CHARMM phosphoinositide templates
match the registered proton location: POP2/PAPI/SAPI use the PI(4,5)P2 P4-protonated
model, and SOP2 uses PI(3,4)P2 protonated on P3, all with charge -4.

LYSPG uses 3′-O-L-lysyl substitution of natural PG, preserving its headgroup
configuration, as defined by [EC 2.3.2.3](https://enzyme.expasy.org/EC/2.3.2.3).
For TMCL and TOCL, identical phosphatidyl arms make the central glycerol carbon
non-stereogenic. Stereochemical completeness checks normalize equivalent anionic
phosphate resonance representations; genuine carbon asymmetry remains checked.
These identity and parameter checks authorize construction candidates, not a
claim of equilibrated membrane properties. Each candidate still requires its
own simulation and construction acceptance before becoming selectable.

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

Administrators can inspect the global library with
`gmxbuilder lipid-library status`. Short `--test-mode` outputs are
smoke-test material and do not pass the production runtime gate.

### Task-private custom lipids

Task-private custom lipid submission is retired in the Web interface. Ask the
administrator to curate a lipid into the shared library. Do not interpret the
legacy identity check as a guarantee of stereochemical equivalence. Library
identity, parameter completeness and physical validation must be established
separately before a new lipid is accepted.

### What a successful construction establishes

Coordinate/topology consistency and successful GROMACS preprocessing are
construction checks. They do not independently establish correct chemical
identity, complete bonded parameters, conversion equivalence or equilibrium
sampling. Pre-equilibrated single-lipid conformers do not establish that an
assembled mixed bilayer, protein complex or solvated system is equilibrated.
The complete system requires its own relaxation, equilibration and assessment
of convergence for the intended observable. Geometric previews are estimates,
not simulation results.

## 4. Small molecules

- GAFF2 requires the user to confirm the integer net charge of every retained
  molecule. Automated suggestions do not replace judgement about pH,
  tautomers, salt form, or coordination state.
- GAFF2 may add hydrogens, but parameterization must preserve the input heavy-
  atom identity and order.
- CHARMM defaults to the [local compatibility backend](charmm_compat/README.md).
  Exact supported templates and explicitly acknowledged research domains have
  different validation scopes. CGenFF import is an alternative and requires
  MOL2 and STR output for the same chemical structure. High-penalty parameters
  require external quantum-chemical validation or refitting.
- Metal coordination, covalent ligands, reactive intermediates, and coupled
  protonation are not general automated parameterization capabilities.

## 5. Nucleic acids

Canonical linear DNA and RNA are supported only by the Solvator workflow, with
one of two force fields:

| force field | DNA | RNA | small molecules |
|---|---|---|---|
| `charmm36m` | CHARMM36 | CHARMM36 | Local CHARMM or matching CGenFF import |
| `amber14sb_ol24` | OL24 | ff99bsc0+χOL3 | **GAFF2 / AM1-BCC, automatic** |

GMXBUILDER treats each chain as a polymer and uses the force field's own
residue databases to construct 5′/3′ hydroxyl termini, hydrogens, O3′–P links,
bonded terms, and an integral chain charge. Protein–DNA, protein–RNA, and
compatible non-covalent ligand complexes are supported. This native
preparation replaces the uploaded nucleic-acid coordinates with the
hydrogen-complete `pdb2gmx` coordinates; the Step 3 viewer is the required
coordinate review point.

**`amber14sb_ol24` is assembled during installation, not shipped.**
`install-local.sh` downloads the Olomouc OL24 package, verifies its pinned
SHA-256, and merges only its nucleic-acid half onto the bundled ff14SB. If the
force field is not offered, run the installer. It requires GROMACS 2026 or
later, inherited from the ff14SB port it is built on.

**For anything that is not a nucleic acid, the two force fields are the same
force field.** No parameter that a residue `amber14sb` already supports
differs between them — asserted over every atom type, bond, angle, dihedral
and constraint in the files, not only over a sample. A peptide built with each
gives identical topologies and identical energies in every term.

There is exactly one behavioural difference, and it is in residue naming
rather than in parameters. `amber14sb` defines `RA` as the radium(2+) ion;
`rna.rtp` defines `RA` as adenosine, and a residue name cannot mean two
things. In `amber14sb_ol24`, `RA` is adenosine. GMXBUILDER never places a
radium ion — the ion catalogue offers Na, K, Li, Cs, Ca, Mg, Zn, Cl, Br and I
— so this can only matter for an uploaded structure containing a residue
literally named `RA`, where adenosine is almost certainly what was meant.

Choose the nucleic-acid model for the intended scientific system and comparison
literature. The installed `forcefield.doc` records the model references. Availability
of both options does not establish equivalent predictions or select a universally
preferred model.

Backbone discontinuities, circular chains, covalent DNA/RNA hybrids, and
modified or noncanonical nucleotides are rejected. Membrane-embedded nucleic
acids and Martini nucleic acids are not available. Nucleotide-like free
ligands remain in the small-molecule workflow instead of being silently
attached to a polymer.

## 6. Protein protonation, termini, and modifications

PROPKA output is a discrete suggestion for one static structure, not constant-
pH MD. It does not jointly solve membrane potential, ligand protonation, metal
coordination, or multiple conformations. Catalytic sites, buried hydrogen-bond
networks, and cofactors require manual review.

Free protein termini are currently represented by the standard NH3+/COO−
templates; no neutral terminal microstate is implemented. The bounds are
derived from the builder's own model pKa values (N-terminus 8.0, C-terminus
3.5) rather than chosen, and fall into three bands. Between pH 4.45 and 7.05
both canonical states are at least 90% populated and the build is silent.
Between pH 3.50 and 8.00 the canonical state is still the majority species:
the build continues and reports the actual population, because molecular
dynamics must assign one discrete state per titratable group and the
convention is to assign the dominant one. Beyond the pKa values the canonical
charged form would be the minority species, which makes the assignment wrong
rather than approximate, so the build stops. Capping changes the chemical construct and is appropriate only when that
construct actually has the specified caps. It is not a substitute for an
unsupported neutral free terminus or a workaround for a rejected pH.
GMXBUILDER does not silently claim unsupported
free-terminal microstates. ACE/NME caps are inserted only when the selected
force field contains an atom-complete template. Residue
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
see the [User Manual](USER_MANUAL.md) for operation.

## 8. Build quality and responsibility

A successful GMXBUILDER build means that the input, coordinate checkpoints,
topology, box, index, and MDP files passed the current automated checks and can
enter minimization and equilibration. It does not establish that every mixture,
temperature, phase, pH, conformation, or modification is experimentally
correct, nor that production sampling has converged.

Atomistic protocols without hydrogen-mass repartitioning or virtual sites are
limited to 2 fs with supported bond constraints and 1 fs with unconstrained
bonds. Advanced MDP overrides are checked after merging into the final rendered
file, so they cannot bypass timestep, constraint, CHARMM non-bonded, cutoff, or
periodic-box rules. The shortest periodic box height must exceed twice the
largest neighbor/electrostatic/van-der-Waals cutoff, and solute-to-face
clearance must cover that cutoff (Z only for a periodic membrane). Generated
velocities use a positive, domain-separated seed recorded in the MDP and
package provenance; `gen-seed=-1` is not emitted.

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


### Parameter provenance and diagnostic limits (0.9.124)

PPCPL/PPEPL use the original West 2020 PLA18 model, adapted by shortening the
vinyl-ether chain to P-16:0 and, for PPCPL, transferring the native POPC headgroup.
The source is [the author's original ether stream](https://terpconnect.umd.edu/~jbklauda/download/toppar_all36_lipid_ether_AL.str),
SHA-256 `67d3f8090ff58c29b4c6c58bd265b10fddf68ac4e8ed3216d58be6995750cb0f`,
DOI [10.1021/acs.jpcb.9b08850](https://doi.org/10.1021/acs.jpcb.9b08850).
Version 0.9.124 corrects C23/H3R/H3S versus C33/H3X/H3Y charge assignment;
old CHARMM PPCPL/PPEPL trajectories require reconstruction with the corrected
parameters. A zero total charge does not establish correct atomic charges.
The later `toppar_all36_lipid_ether.str` contains different parameters and is
not silently substituted for the named original model.

Experimental CHARMM models remain opt-in. Matching an environment or reaching
its maximum search radius is not independent corroboration. A single charge
witness does not measure uncertainty. The linear model's two held-out homologues
do not establish validity across independent chemical scaffolds or physical
observables. Charge means and distributed total-charge correction are explicit
approximations, not a new validated force field.

V4 local-conformation admission, area stationarity and physical phase validation
are distinct. Existing protocols retain their recorded temperature (including
the 308.15 K floor and 315.15 K host condition), reference temperature and
adjustment flag; a new temperature requires a new protocol fingerprint.
Comparison scripts report area diagnostics, not a substitute production gate.
CG placement buffers, random kicks and generic thicknesses are construction
settings; they are not independently calibrated material properties.

Ion reports distinguish requested concentration from count-derived concentration.
The latter includes neutralization ions and uses the initial replaceable-water
count times 0.0299 nm³ as its solvent-volume estimate. This is not an equilibrium
measurement, and rounding to whole ion counts remains visible.

New automatic GAFF installations use an exact Linux x86_64 artifact lock for
AmberTools 24.8 / ACPYPE 2023.10.27 / Open Babel 3.1.1. The formerly declared
AmberTools 26.0 combination cannot resolve with this Open Babel version through
the reviewed conda-forge dependencies. Existing environments are not overwritten;
other platforms require a separately validated runtime until a reviewed lock exists.


DOPGD/DPPGD atomic types and charges were compared atom-for-atom with the
Wu2014 DODG/DPDG stream records preserved by Limonada (DOI
[10.1002/jcc.23702](https://doi.org/10.1002/jcc.23702)). Both match; their values
are unchanged. Source SHA-256 values are
`249e6704b0b17c6812b05836537d3d14c62ac6a22f332a1751f2da4632c5d591` (DODG)
and `d97e91ce270da8459fefd60c73382dc471e0a69539d534e3f1d68dc177126251` (DPDG).
This establishes conversion provenance, not validation against later reparameterizations
or proof of physical accuracy in every mixture.
