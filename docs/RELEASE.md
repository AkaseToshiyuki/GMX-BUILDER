# Download and install GMXBUILDER 1.0.0

<p><strong>English</strong> · <a href="RELEASE.zh-CN.md">简体中文</a></p>

The [1.0.0 release](https://github.com/AkaseToshiyuki/GMX-BUILDER/releases/tag/v1.0.0)
contains the Python wheel, source distribution, English and Chinese PDF manuals,
release notes and `SHA256SUMS`. The tagged source is the recommended starting
point for the complete scientific installation:

```bash
git clone --branch v1.0.0 --depth 1 https://github.com/AkaseToshiyuki/GMX-BUILDER.git
cd GMX-BUILDER
./install-local.sh
```

The installer can fetch the manifest-pinned lipid archive over HTTPS without Git
LFS or an access token. To hydrate all LFS files in a source checkout explicitly,
install Git LFS and run `git lfs install` and `git lfs pull` inside the checkout.
See the [user manual](USER_MANUAL.md) for bootstrap dependencies, server modes,
resource limits and the manual installation path. The fully locked GAFF runtime
currently targets Linux x86-64. Managed Web storage also requires FUSE support.

## Files and directories

| Download | Purpose |
| --- | --- |
| `gmxbuilder-1.0.0-py3-none-any.whl` | Python package and bundled asset v7 |
| `gmxbuilder-1.0.0.tar.gz` | Source distribution with installer, scripts and user documentation |
| `USER_MANUAL.pdf`, `USER_MANUAL.zh-CN.pdf` | Versioned release manuals |
| `RELEASE_NOTES_1.0.0.md`, `RELEASE_NOTES_1.0.0.zh-CN.md` | Features and scientific limits |
| `SHA256SUMS` | SHA256 of the downloadable files |

Verify downloaded files with `sha256sum -c SHA256SUMS` from their directory.
A wheel installs Python code and packaged data; it does not by itself install
GROMACS, AmberTools, system/FUSE dependencies or separately licensed force fields.
Use the matching source installer to prepare those dependencies. GitHub's automatic
source ZIP/tar download may contain LFS pointers; use a hydrated Git clone or the
attached source distribution for an offline copy of the bundled assets.

The source tree keeps code under `src/gmxbuilder/`, user documentation under
`docs/`, installation helpers under `scripts/`, and service examples under
`deploy/`. Local release builds go into `dist/1.0.0/`. Runtime caches and tasks
are stored separately from this release directory; consult the manual before
changing storage locations. Installation does not run MD.

See the [release notes](RELEASE_NOTES_1.0.0.md), [lipid support matrix](release-support.md),
[scientific limits](SCIENTIFIC_COMPATIBILITY.md) and [third-party notices](../THIRD_PARTY_NOTICES.md).
