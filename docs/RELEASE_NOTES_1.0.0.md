# GMXBUILDER 1.0.0

<p><strong>English</strong> · <a href="RELEASE_NOTES_1.0.0.zh-CN.md">简体中文</a></p>

Released 2026-10-08. GMXBUILDER provides five guided GROMACS construction
workflows: atomistic protein–membrane systems, pure bilayers, solvated systems,
Martini 3 bilayer/protein–membrane systems and Martini 3 solvated proteins.

- Web, CLI and HTTP API interfaces support capability discovery, staged inspection,
  final structure confirmation and downloadable simulation input packages.
- Managed Web execution supports asynchronous operations, task recovery,
  bounded queues and configurable resource limits. Local, authenticated public
  and anonymous public deployments have explicit access policies.
- Asset v7 distributes 37 accepted lipid/backend initialization libraries,
  134,000 conformers and 67 prepared GAFF2/AM1-BCC parameter caches.
- Exact Lipid21, GAFF2, local CHARMM/CGenFF and Martini routes enforce their
  molecular identity, parameter and force-field compatibility contracts.
- English and Chinese manuals document installation prerequisites, current API
  behavior, generated files and scientific limitations.

This release supports the combinations listed in the [support matrix](release-support.md).
The other 299 registered lipid/source combinations are not distributed as accepted
libraries. Initialization conformers do not establish membrane equilibrium,
physical validity of a new mixture or production convergence. Protein, nucleic
acid, ligand, custom-lipid and research-model limitations remain explicit in the
[scientific guide](SCIENTIFIC_COMPATIBILITY.md). GROMACS preprocessing and software
regression results do not replace system-specific physical validation.

Use the [download and installation guide](RELEASE.md). Upgrades preserve newer
valid library caches; check actual installed availability after upgrading.
Installation starts no molecular dynamics or offline library-production queue.
