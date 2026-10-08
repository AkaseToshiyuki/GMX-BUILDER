# Local CHARMM-compatible ligand assignment

In the force-field step, CHARMM36/CHARMM36m select **Local CHARMM-compatible
builder** by default. Each retained ligand defaults to **Identify automatically**.
The interface displays its chemical identity, net charge and information source.
Use **Provide MOL2** or **Provide SMILES** when identification needs clarification,
or to override the proposed chemical state. The alternative **CGenFF / ParamChem import**
still accepts the matched MOL2 + STR package. Selecting an installed family is
not a claim that every molecule is covered; Check evaluates the chemistry.

This is an independent implementation, not official CGenFF output. It is **not
currently a general replacement for CGenFF**. Complete-molecule recognition and
new-scaffold assignment have different coverage; see [COVERAGE.md](COVERAGE.md).
A residue name such as AMP never establishes chemical identity.

## Automatic chemical identification

The resolver first uses embedded mmCIF atom/bond definitions or an original
hydrogen-complete PDB ligand, then a matching wwPDB CCD definition, then bounded
Open Babel coordinate perception. Original uploads remain the chemical source
even when the working coordinates have been filtered, translated or rotated.
CCD IDs are only lookup hints: elements, connectivity, atom mapping and stereo
must agree with the actual retained ligand. CCD lookups fetch a public entry by
component ID; no molecular coordinates/files are submitted. Definitions are
cached under `~/.cache/gmxbuilder/ccd` (`GMXBUILDER_CCD_CACHE` overrides the root).
An unavailable or mismatched CCD entry falls through to coordinate perception.

Automatic chemistry uses Open Babel's solution-pH model at the requested pH
and pH ±0.5 (clamped to 1–13). Different resulting states require clarification.
Coordinate-only inputs with multiple enumerated tautomers, ambiguous graph maps,
undefined/inconsistent stereochemistry, malformed valence, disconnected atoms
or explicit PDB links to another residue/metal are not silently accepted.
These gates do not constitute a universal ambiguity detector or bound-state pKa
prediction. The interface identifies coordinate-derived results as predictions.

Explicit MOL2/SMILES overrides take priority and are not re-protonated by pH.
MOL2 supplies chemical bonds/types and the intended state; its partial charges
are ignored. Heavy-only MOL2 may use Open Babel to complete hydrogens from its
supplied types/bonds, without a pH override. Matching atom names resolve atom
identity; otherwise only chemically equivalent graph mappings are accepted.
SMILES must include charge/stereo; numbered maps, when needed, label all heavy
atoms 1..N in retained coordinate order. The original heavy coordinates are
preserved exactly. Uploaded MOL2 coordinates never replace the bound pose.

Identification previews and Check share the resolver. Check repeats the mapping
against every actual ligand instance, then runs the existing CHARMM parameter
coverage/completeness checks. Recognizing a new molecule does not extend the
force-field model's domain. A changed pH, input mode or task invalidates old
preview responses. MOL2 files are validated before saving; a failed replacement
cannot silently reuse the previously selected file. Resume restores input modes,
SMILES overrides and the selected validated MOL2.

Automatic perception needs Open Babel (the existing managed GAFF environment's
`obabel`, or one on PATH); no AmberTools charge calculation is invoked. Explicit
SMILES and hydrogen-complete MOL2 do not require that tool. CLI/module callers
may omit `charmm_compat_smiles` for automatic identification, or provide
`charmm_compat_mol2: {LIG: /path/to/ligand.mol2}`. Web clients use task-owned upload
references, never arbitrary host paths. Assignment JSONs retain identity source,
SMILES, atom maps, pH policy, source hash (when available), and model warnings.

References: [wwPDB CCD](https://www.wwpdb.org/data/ccd),
[Open Babel conversion/protonation options](https://openbabel.org/docs/Command-line_tools/babel.html),
[RDKit coordinate connectivity](https://www.rdkit.org/docs/source/rdkit.Chem.rdDetermineBonds.html).

## Complete templates and research assignment

The catalog contains **61 reviewed identities** spanning alcohols/ethers,
amines/ammonium ions, aromatics, aldehydes/ketones, nitriles, amides/urea,
esters, acids/carboxylates, N/O/S heterocycles and phosphate states. These are
specific complete molecular states, not generic typing rules for every molecule
containing those functional groups. All types/charges and parameters come from
the selected installed release; no parameter values are embedded in the catalog.
Native generation, coordinate mapping and numerical/preprocessing checks still
gate every actual build. Version-specific exclusions are in the coverage table.

The existing `linear-neutral-co-bci-v1` research model covers neutral unbranched
acyclic saturated C/O molecules (one oxygen, 24 heavy atoms). Its eight fitted
bond increments reproduce the training MEOH/ETOH/PRPA/DMEE/DETE references and
held-out PROH/METE homologues. This is reference reproduction, not physical QA.

The `aryl-primary-ammonium-tied-bci-v1` model covers saturated acyclic C/H
skeletons, optionally one monosubstituted benzene and one primary alkyl NH3+
group, up to 24 heavy atoms, with explicit stereochemistry. Unsupported local
type environments or missing parameters still block export. The reported AMP
is S-amphetamine at +1, supplied as `C[C@H]([NH3+])Cc1ccccc1`. Its 24-atom model
has been constructed under both releases. This does not imply support for
neutral amphetamine, secondary amines, arbitrary heterocycles, fused-ring new
scaffolds, polyfunctional drugs or covalent/metal-bound ligands.

Types are learned from reviewed local valence environments in MAMM, EAMM, BENZ,
EBEN and CUME. ALAI supplies the ammonium-adjacent CH type, **not its zwitterionic
charges**. Six identifiable bond-transfer coefficients are fitted to the five
training references. Formal atomic charge is the baseline; antisymmetric
transfers conserve the exact molecular charge without uniform corrections.
CH/CH2/CH3 substitution states share declared charge classes. Adjacent-N carbon
hydrogen transfers remain distinct and are fitted separately for each release.
AMP's alpha CH and its hydrogen are explicitly reported as charge-environment
extrapolations. This class sharing is a research hypothesis, not an AMP QM fit.

TOLU, BZAM and PRPA are withheld from typing/charge training. Toluene/propane
charges reproduce, but benzylammonium has maximum errors of **0.09 e in 4.6**
and **0.13 e in 4.1** (RMSE 0.0260/0.0407 e). These nonzero errors expose missing
longer-range polarization. They are recorded as diagnostic errors, not an
accuracy acceptance test or an official CGenFF penalty. Independent QM,
interaction-energy, torsion-profile and condensed-phase validation remain open.

Bonded lookup first retains native exact/wildcard terms. Only the new research
model can fill a missing term with a documented **single substitution**:
ammonium-adjacent CH/CH2/CH3 to its neutral aliphatic counterpart, or neutral
CH3 to CH2 at a proper-torsion end. Terminal substitutions are preferred to
central substitutions; equally ranked conflicting candidates block assignment.
Each analogy must land on an exact tuple in the same installed release.
All Fourier terms and Urey-Bradley values are copied, with file/line provenance;
LJ, atom types and charges are not changed by bonded analogy. This is an
uncalibrated research transfer, never silently applied to complete templates.
No general substitution search or invented force constants are used.

The public algorithmic reference is [CGenFF Automation II](https://doi.org/10.1021/ci3003649).
It describes a broader charge-increment/analogy approach; this implementation
neither reproduces the official rule base nor establishes equivalent accuracy.

## Solution pH and explicit molecular state

The force-field panel now exposes **Solution pH for ligand protonation**, 1–13.
It shares the value with protein processing. Editing either control invalidates
previous checks and computed ligand suggestions. GAFF2 suggestions and actual
protonation receive this pH; the existing GAFF cache already includes it in its
identity. Late responses from an old pH/task cannot replace newer suggestions.
Explicit integer-charge overrides are retained for review. Invalid/blank pH
blocks Check instead of silently reverting to 7.0. Resume restores saved
`forcefield.ligand_pH`, with the older protein pH as fallback.

Automatic CHARMM identification applies the pH model described above before
force-field assignment. Explicit MOL2/SMILES and imported CGenFF packages retain
their supplied state; for these inputs pH records the intended environment.
CHARMM atomic partial charges still come from its selected template or research
model, never from Open Babel or GAFF. There is no universal local pKa predictor.

## Version, mapping and validation

CHARMM36 uses the existing **Mar2019 / CGenFF 4.1** `merged.rtp`/`merged.hdb`;
CHARMM36m uses **Jul2022 / CGenFF 4.6** `cgenff.rtp`/`cgenff.hdb`. Each fit and
assignment uses only its own selected release. No database upgrade or parameter
download is performed by assignment. Source files are hashed and the completed
ITP/atomtypes files are also hashed. Export rejects missing reports, altered
artifacts or a different force field. Existing installation-source checksums
and upstream data notices remain in `scripts/external_assets.json` and
`THIRD_PARTY_NOTICES.md`; the project grants no new data redistribution rights.

The resolved or supplied SMILES establishes chemistry before parameter assignment.
Automatic bond-order perception belongs to the separate identification stage and
is reported as such. Atom counts/elements/connectivity must agree. Undefined stereocenters and coordinate/SMILES stereochemical disagreement are
rejected. Multiple mappings that assign different parameters require explicit
atom-mapped SMILES (numbers 1..N are retained PDB heavy-atom order). For example,
`[CH3:1][C:2](=[O:3])[OH:4]` identifies the carbonyl and hydroxyl oxygen when
those atoms are the first through fourth heavy atoms. This records user-supplied
identity; parameter assignment does not alter that state. Conflicting maps fail. Equivalent mappings are
resolved by stable original atom names. Export's shared external-ligand guard
requires every instance's coordinate atom order to match its ITP.

Native [GROMACS pdb2gmx](https://manual.gromacs.org/current/onlinehelp/gmx-pdb2gmx.html)
owns topology generation, exclusions, 1–4 terms, impropers and multi-term proper
torsions. Research analogies are resolved explicitly before label adaptation; the selected base force field retains
LJ, pair types and NBFIX. An [OpenMM GromacsTopFile](https://docs.openmm.org/latest/api-python/generated/openmm.app.gromacstopfile.GromacsTopFile.html)
static energy/force comparison checks native versus adapted topology at identical
coordinates on the Reference platform (absolute tolerances 1e-6 kJ/mol and
1e-5 kJ/mol/nm). This is a conversion check using one engine, not an independent
cross-engine force-field benchmark. Native `grompp -maxwarn 0` must also pass.
No integrator steps, GROMACS minimization or MD trajectory are executed.

`topology/<ligand>_assignment.json` travels in the downloadable ZIP. It contains
source/model/software identities, charge contributions for research assignments,
mapping, input/artifact hashes, numerical status and the separate physical
validation status. Experimental exports also identify that status in README
and topology headers. Task artifacts stay under the task's `charmm_compat/`
directory; command-line builds use the user's private cache.

## Inspection and configuration

```bash
gmxbuilder charmm-compat doctor --ff charmm36m
gmxbuilder charmm-compat inspect --ff charmm36m --smiles CCCCO
```

`inspect` produces a research report without coordinates or simulation files.
It does not certify topology completeness before the actual coordinate-based
Check. A build configuration uses:

```yaml
modules:
  forcefield:
    name: charmm36m
    ligand_ff: charmm_compat
    ligand_pH: 7.4
    charmm_compat_smiles:
      LIG: CCCCO
    charmm_compat_allow_research: true
```

Keep the research flag false for complete-template-only assignment. Unsupported
molecules remain available through the existing CGenFF import path.
