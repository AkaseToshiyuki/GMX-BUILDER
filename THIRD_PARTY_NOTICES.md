# Third-Party Notices

<p><strong>English</strong> · <a href="THIRD_PARTY_NOTICES.zh-CN.md">简体中文</a></p>

GMXBUILDER combines original orchestration code with scientific data formats,
force-field ports and optional external programs. The project-level
GPL-3.0-or-later license applies only to original GMXBUILDER code and
documentation. It does
not replace upstream licenses, citation requirements or usage conditions.

## Force-field data

The public source distribution includes the GROMACS-provided Amber and OPLS
data listed below together with their provenance files. It does not redistribute
CHARMM36/CHARMM36m. Before Python dependency installation, the local installer
downloads those two pinned GROMACS ports directly from the official MacKerell
Lab endpoint, verifies SHA-256, applies the documented GMXBUILDER overlays, and
installs them locally.

Every non-redistributed runtime asset is enumerated in
`scripts/external_assets.json` with its official source page, direct HTTPS
download URL, archive root, required files and pinned SHA-256 digest. The
unattended `install-local.sh` invokes that manifest before installing Python
dependencies; no manual browser download or file placement is required.

- `src/gmxbuilder/data/forcefields/amber14sb.ff`
- `src/gmxbuilder/data/forcefields/amber99sb.ff`
- `src/gmxbuilder/data/forcefields/charmm36m` (installed locally)
- `src/gmxbuilder/data/forcefields/charmm36` (installed locally)
- `src/gmxbuilder/data/forcefields/oplsaa.ff`

Provenance and scientific references embedded in `forcefield.itp`,
`forcefield.doc` and related source headers must be preserved. Before any
redistribution, distributors must verify the upstream terms for every bundled
force-field file. This notice does not grant redistribution rights. Public
deployment must use `scripts/install_external_assets.py`; changing a pinned
source or checksum is a reviewed scientific change, not an automatic upgrade.

## Prebuilt lipid assets

`src/gmxbuilder/data/prebuilt_assets/` contains generated simulation outputs.
The installer can hydrate its Git LFS payload directly from the manifest-pinned
public HTTPS media URL, so end users do not need Git LFS or an access token.
The payload contains:

- explicit-solvent, semi-isotropic NPT lipid conformations;
- GROMACS topology/cache files generated through GAFF2 and AM1-BCC tooling.

The archive contains no AmberTools, ACPYPE or GROMACS executable code. These
programs remain external dependencies. The Amber project identifies its force
fields as public-domain material; the cached topology and coordinate files are
GMXBUILDER-generated outputs rather than copies of AmberTools program source.
Generated parameters still require the appropriate force-field citations and
scientific validation for the intended system.

AmberTools is described by its authors as mostly GPL-licensed, with component-
specific terms, while the Amber force fields are described as public domain.
Consult the official Amber pages and distribution license files:
<https://ambermd.org/AmberTools.php> and <https://ambermd.org/>.

The GROMACS executable is installed separately. Force-field files copied from
the GROMACS distribution remain under the upstream LGPL-2.1-or-later terms;
the retained license text is stored at
`src/gmxbuilder/data/forcefields/LICENSE-GROMACS-LGPL-2.1.txt`. GROMACS license
and citation guidance are available from <https://www.gromacs.org/>.

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
project GPL license does not replace Martini model citation requirements.

## Installed Python dependencies

The installer resolves exact artifacts from `uv.lock`; dependency source code
is not copied into this repository. Major scientific dependencies declare the
following licenses in their installed package metadata. Their complete license
texts and any bundled-component notices remain in the installed distributions.

| Dependency | Declared license |
|---|---|
| NumPy / SciPy | BSD-3-Clause family, with NumPy bundled-component notices |
| RDKit | BSD-3-Clause |
| OpenMM | BSD-like |
| PDBFixer | MIT |
| Vermouth / Martinize2 | Apache-2.0 |
| COBY | Apache-2.0 |
| MDTraj | LGPL-2.1-or-later |
| Matplotlib | PSF-compatible Matplotlib license |

The remaining web and packaging dependencies are also installed from pinned
artifacts and retain their own upstream metadata. Repackaging the complete
virtual environment requires preserving every dependency notice; this project
does not grant replacement terms.

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
