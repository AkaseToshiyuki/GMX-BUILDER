# Third-Party Notices

<p><strong>English</strong> · <a href="THIRD_PARTY_NOTICES.zh-CN.md">简体中文</a></p>

GMXBUILDER combines original orchestration code with scientific data formats,
force-field ports and optional external programs. The project-level MIT
license applies only to original GMXBUILDER code and documentation. It does
not replace upstream licenses, citation requirements or usage conditions.

## Force-field data

The directories below contain force-field data or converted ports:

- `src/gmxbuilder/data/forcefields/amber14sb.ff`
- `src/gmxbuilder/data/forcefields/amber99sb.ff`
- `src/gmxbuilder/data/forcefields/charmm36m`
- `src/gmxbuilder/data/forcefields/charmm36`
- `src/gmxbuilder/data/forcefields/oplsaa.ff`

Provenance and scientific references embedded in `forcefield.itp`,
`forcefield.doc` and related source headers must be preserved. Before any
distribution outside a private research repository, the repository owner must
verify the upstream redistribution terms for every bundled force-field file.
This notice does not grant redistribution rights.

## Prebuilt lipid assets

`src/gmxbuilder/data/prebuilt_assets/` contains generated simulation outputs:

- explicit-solvent, semi-isotropic NPT lipid conformations;
- GROMACS topology/cache files generated through GAFF2 and AM1-BCC tooling.

The archive contains no AmberTools, ACPYPE or GROMACS executable code. These
programs remain external dependencies. Generated parameters still require the
appropriate force-field citations and scientific validation for the intended
system.

AmberTools is described by its authors as mostly GPL-licensed, with component-
specific terms. Consult the official Amber distribution and license files:
<https://ambermd.org/AmberTools.php>.

GROMACS is not bundled. Its license and citation guidance are available from:
<https://www.gromacs.org/>.

## Martini 3 coarse-grained workflow

`src/gmxbuilder/data/martini3/` contains pinned Martini 3 interaction and
molecule topology files from the Martini Force Field Initiative. The bundled
manifest records exact upstream commits and SHA-256 values. The retained
upstream `LICENSE.txt` is Apache-2.0 and must remain with redistributed files:
<https://github.com/Martini-Force-Field-Initiative/M3-Lipid-Parameters>.

The independent coarse-grained workflow depends on pinned external Python
packages rather than vendoring their source:

- Vermouth/Martinize2 0.15.0 for protein mapping;
- COBY 1.0.14 for flat membrane/solvent/ion assembly;
- MDTraj 1.10.3 as a coordinate-processing dependency.

Their respective licenses and citation requirements remain upstream. Generated
packages include `CITATIONS.json` with the model and tool references. The
project MIT license does not replace Martini model citation requirements.

## User-supplied parameter files

External small-molecule MOL2/STR packages uploaded by users are stored only
under the corresponding task directory and are not part of the repository or
prebuilt release archive.

## Browser visualization libraries

GMXBUILDER distributes pinned local browser builds so the Viewer does not
execute mutable CDN code:

- 3Dmol.js 2.5.5, BSD-3-Clause, under
  `src/gmxbuilder/web/static/vendor/3dmol-2.5.5/`;
- SmilesDrawer 2.0.3, MIT, under
  `src/gmxbuilder/web/static/vendor/smiles-drawer-2.0.3/`.

The corresponding upstream license files are retained beside each minified
asset. `src/gmxbuilder/web/static/vendor/ASSET_MANIFEST.json` records the npm
package integrity and installed-file SHA-256 values used by release checks.
