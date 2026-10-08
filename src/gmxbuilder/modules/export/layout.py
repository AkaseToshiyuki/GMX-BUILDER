"""Paths inside a self-contained simulation export.

The root holds README.txt, CITATIONS.json, manifest.json and run_md.sh.
Coordinates and index groups live in structure/, topology in topology/,
the bundled force field in topology/forcefield/, and stage settings in mdp/.

Topology includes must resolve within the package from the including file's
directory. Preserve that relative layout when moving or archiving an export."""

from __future__ import annotations

#: Coordinates and the index groups that address them.
STRUCTURE_DIR = "structure"

#: ``topol.top`` and every ``.itp`` it includes that describes *this* system.
TOPOLOGY_DIR = "topology"

#: The bundled force-field database, copied verbatim. Nested inside the
#: topology directory so that every include in the package points downward.
FORCEFIELD_DIR = "forcefield"

#: Generated MD parameter files, one per stage.
MDP_DIR = "mdp"

#: Files that stay in the package root, in the order a reader wants them.
ROOT_FILES = (
    "README.txt",
    "manifest.json",
    "CITATIONS.json",
    "run_md.sh",
)

#: Written by earlier versions into the package root. An export directory is
#: reused across reruns, so these are removed before writing the sorted layout
#: -- otherwise a stale flat copy sits beside the new sorted one and the user
#: cannot tell which topology `grompp` actually read.
LEGACY_ROOT_ARTIFACTS = (
    "input.gro",
    "input.pdb",
    "index.ndx",
    "topol.top",
    "run_md.sh",
)

#: Suffixes of the force-field database files copied verbatim by the topology
#: writer. Anything with one of these suffixes in a legacy flat export root is
#: a stale copy of a file that now lives under ``topology/``.
LEGACY_TOPOLOGY_SUFFIXES = (".itp", ".rtp", ".atp", ".hdb", ".tdb", ".arn", ".r2b")
