<p align="center">
  <img src="src/gmxbuilder/web/static/assets/gmxbuilder-logo.png" alt="GMXBUILDER logo" width="520">
</p>

<h1 align="center">GMXBUILDER</h1>

<p align="center"><strong>English</strong> · <a href="README.zh-CN.md">简体中文</a></p>

**Current stable release: 1.0.0.** [Release files and installation layout](docs/RELEASE.md).

GMXBUILDER is a Web, command-line, and HTTP API application for preparing
checkpointed GROMACS simulation packages. It supports atomistic membrane,
pure-bilayer, and solution systems, together with dedicated Martini 3 bilayer
and solution workflows. A successful build produces coordinates, topology,
simulation parameters, a run script, a manifest, and method citations.

<p align="center">
  <img src="docs/architecture.svg" alt="GMXBUILDER workflow overview" width="900">
</p>

## Workflows

| Workflow | Supported system |
|---|---|
| Bilayer Builder | Atomistic membrane protein, mixed or pure bilayer, water, and ions |
| Pure Bilayer System | Protein-free atomistic bilayer, optionally without solvent |
| Solvator | Atomistic protein, supported canonical DNA/RNA, and compatible non-covalent ligands in solution |
| Martini 3 Bilayer Builder | Standard coarse-grained protein in a flat symmetric or asymmetric bilayer |
| Martini 3 Solvent Builder | Standard coarse-grained protein in water and ions |

Every **Check** saves the exact coordinates shown in the Viewer. Later steps
consume that checkpoint, and Build packages the final confirmed system without
re-running coordinate construction. Unsupported force-field combinations,
chemical identities, modifications, and molecular classes are reported rather
than silently approximated.

## Quick start

### Lipid asset scope

Asset release **v7** distributes **37 accepted lipid/backend combinations** and
134,000 initialization conformers. Its 67 GAFF2 parameter caches are separate
from conformer acceptance: a fitted cache does not make a lipid available for
membrane construction. See the [supported combinations and validation scope](docs/release-support.md).
Pending or incompatible combinations remain unavailable; full V4 library
coverage is not a prerequisite for using the accepted subset.

These assets support initial construction and do not certify membrane equilibrium
or production sampling. Each newly constructed system still needs equilibration.

### Installation

Bootstrap requirements:

- Linux x86-64 and Python 3.10 or later;
- CMake, a C++17 compiler, and Python's `venv` module;
- Internet access during first installation;
- systemd, cgroup v2, Landlock ABI 3+ and FUSE3 for managed Web operation; see [installation requirements](docs/USER_MANUAL.md#21-bootstrap-requirements);
- the NVIDIA CUDA toolkit only for a CUDA-accelerated managed GROMACS build.

Clone the public repository and run the installer:

```bash
git clone --branch v1.0.0 --depth 1 https://github.com/AkaseToshiyuki/GMX-BUILDER.git
cd GMX-BUILDER
./install-local.sh
```

The installer reuses a compatible GROMACS 2026.0-or-newer executable or builds
the verified official GROMACS 2026.3 source locally, installs a managed
GAFF2/AM1-BCC runtime, retrieves separately distributed force-field data and
the prebuilt lipid archive from pinned HTTPS sources, verifies SHA-256 digests,
installs Python dependencies, populates the user cache, and starts the local
service. Git LFS, root access, and a GitHub access token are not required.
The default invocation is unattended and uses safe local settings. Use
`./install-local.sh --help` for explicit address, port, CPU, queue, or optional
interactive configuration.

The service defaults to `127.0.0.1:7788`. In `local` / `trusted-lan` mode,
unauthenticated non-loopback listeners require explicit `--allow-unsafe-deployment`
and a protected private network. Public hosting uses either authenticated
`public` mode or resource-isolated `public-anonymous` mode; both require a trusted
TLS proxy and explicit HTTPS origins. See [deployment configuration](docs/USER_MANUAL.md#appendix-a-deployment-configuration).

Open <http://127.0.0.1:7788/>. Save the displayed Task ID; it is the only key
needed to resume an unexpired task or download a completed package again.

## Command line and API

Inspect the installed capabilities before choosing a force-field combination:

```bash
gmxbuilder --version
gmxbuilder list-ff
gmxbuilder list-water
gmxbuilder list-lipids
gmxbuilder lipid-library status
gmxbuilder --help
```

The complete YAML/CLI workflow and examples are documented in the user manual.
When the service is running, request and response schemas are available from
`/docs` and `/openapi.json`; the installed schema is authoritative.

## Output and scientific boundary

A wet-system package uses this directory layout:

```text
README.txt        build settings and results
manifest.json     versions, input hashes, settings and output checksums
CITATIONS.json    references for this system
run_md.sh         launcher
structure/        input.gro, optional input.pdb and index.ndx
topology/         topol.top and included parameters
mdp/              enabled simulation stages
```

Retain the original input and matching parameter sources for reconstruction;
the uploaded input itself is not redistributed. Exact contents depend on the
workflow and composition. Dry bilayer packages omit solvent and simulation stages.

Passing the automated checks means the package is structurally and
topologically ready to enter minimization and staged equilibration. It does not
prove a biological orientation, protonation state, phase, parameter choice, or
converged production trajectory. Review the final coordinates, total charge,
force-field compatibility, and generated citations before simulation.

## Documentation

- [User Manual — 1.0.0](docs/USER_MANUAL.md)
  ([PDF](docs/USER_MANUAL.pdf))
- [Scientific Compatibility and Limitations](docs/SCIENTIFIC_COMPATIBILITY.md)
- [Licensing](LICENSING.md)
- [Third-Party Notices](THIRD_PARTY_NOTICES.md)

## Citation and license

If GMXBUILDER supports published work, cite the software and all method,
force-field, water-model, and parameterization references listed in the
exported `CITATIONS.json`. Repository citation metadata is provided in
[`CITATION.cff`](CITATION.cff).

Original GMXBUILDER code and documentation are licensed under the GNU General
Public License v3.0 or later. Distributed modified versions must remain under
the GPL and provide their corresponding source; proprietary derivatives are
not permitted. Scientific data, force fields, generated parameters, and
external programs retain their upstream licenses and citation requirements.
See the licensing guide and third-party notices before redistribution.
