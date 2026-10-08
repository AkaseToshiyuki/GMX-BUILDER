# GMXBUILDER 1.0.0 lipid support

<p><strong>English</strong> · <a href="release-support.zh-CN.md">简体中文</a></p>

GMXBUILDER **1.0.0** ships the **V4 lipid library**, containing 37 accepted
lipid/backend combinations and 134,000 conformers. The corresponding
[machine-readable matrix](../src/gmxbuilder/data/prebuilt_assets/support-matrix.json)
records each combination, conformer count, and validation scope. Application and
asset versions are separate; installation and runtime admission verify their
compatibility.

V4 names the lipid-library generation. The distribution manifest uses
`asset_version: 7` for bundle revision 7; this does not denote a V7 library.

## Distributed combinations

| Backend | Combinations | Lipids |
| --- | ---: | --- |
| Lipid21 / Amber14SB | 9 | DOPC, DOPE, DOPS, DPPC, DPPE, POPC, POPE, POPS, PSM |
| GAFF2 / Amber14SB | 1 | POP2 |
| CHARMM36m | 14 | CHOL, DOPC, DOPE, DOPS, DPPC, DPPE, POP2, POP3, POPC, POPE, POPG, POPI, POPS, PSM |
| CHARMM36 | 13 | CHOL, DOPC, DOPE, DOPS, DPPC, DPPE, POP3, POPC, POPE, POPG, POPI, POPS, PSM |

These are combinations, not 37 distinct lipid species. Some CHARMM36 entries
reuse coordinates under a verified CHARMM36m parameter-equivalence contract;
they are not independent additional sampling. Mixtures must satisfy every
selected component's parameter/backend contract. A ready entry does not override
an unsupported force-field or mixing combination.

The catalog includes 336 lipid/source combinations. The remaining 299 are not
distributed as accepted conformer libraries in this snapshot. Catalog size is
not a promise that every combination is chemically compatible or will become
supported. A later accepted asset release can expand coverage without waiting
for the entire catalog to complete.

## Scientific scope

All 37 entries are accepted as **initialization conformers**. Recorded identity,
parameter fingerprints, coordinate/topology atom order, geometry, provenance,
independent-replica and local-conformation checks remain attached to each entry.
This does not certify bulk membrane equilibrium, equilibrium conformer weights,
quantitative equilibrium area per lipid, or convergence of a user's production
simulation. Equilibrate every newly constructed system for its own composition
and conditions.

GAFF2 POP2 was sampled in the recorded POPC-host environment. Its availability
does not certify a pure POP2 bilayer's equilibrium properties. The distributed
metadata preserves that sampling environment and its parameter contract.

The archive also includes 67 current GAFF2/AM1-BCC parameter caches. These are
parameter preparation artifacts, not 67 accepted conformer libraries. They do
not unlock pending entries. Assets with missing evidence or incompatible
identity, parameters or protocols remain unavailable rather than falling back
silently to an unvalidated built-in lipid geometry.

## Installation and compatibility

Use the installer to hydrate the archive and acquire separately distributed
force fields, then inspect the installed catalog:

```bash
gmxbuilder prebuilt-assets status
gmxbuilder lipid-library status
```

The asset manifest pins the archive's size and SHA256. The archive contains
conformers, their evidence metadata and prepared GAFF2 caches; separately
distributed force-field databases are not copied into it. Fresh installation
requires those force-field prerequisites as well as the archive. Upgrades
preserve newer valid caches; actual installed availability is authoritative.

The reader recognizes the two specifically reviewed CHARMM installer layouts:
legacy supplemental filenames and their current `gmxbuilder-` names. Only the
pinned file hashes qualify; altered coefficients, missing files and duplicate
old/new supplements do not inherit the old identity. The entrypoint directives
and 36 retained compiled topology/input-parameter sets were checked for equality.
Original evidence fingerprints remain unchanged.

The existing scoped PSM preprocessing policy is limited to its reviewed
128-PSM/TIP3P/equal-NaCl case and GROMACS 2026.3 single precision. Inclusion of
PSM conformers does not permit warnings for arbitrary sizes, mixtures, tool
versions or unrelated charge discrepancies. The user's constructed system must
pass its own preprocessing checks.

## Release boundary

The five application workflows are available within their documented molecular
and parameter limits. This matrix is the distributed atomistic-lipid subset;
Martini 3 exposes its separate capability list. Unsupported combinations remain
unavailable. Construction checks and software tests do not replace physical
validation. Installing 1.0.0 does not start or resume a lipid simulation queue.
