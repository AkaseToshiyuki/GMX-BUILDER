"""Exact installed CHARMM parameters for incremental CGenFF stream imports.

The installed GROMACS port is the version-matched source of truth. Convert
its supported additive terms to CHARMM units for the existing STR emitter;
there is no atom-type substitution or parameter analogy in this path.
"""

from gmxbuilder.core.exceptions import ModuleConfigError
from gmxbuilder.modules.forcefield.catalog import force_field_directory


def _records(path):
    section = ""
    for raw in path.read_text().splitlines():
        code = raw.partition(";")[0].strip()
        if code.startswith("["):
            section = code.strip("[] ").lower()
        elif code and not code.startswith("#"):
            yield section, code.split()


def load_cgenff_base(force_field):
    root = force_field_directory(force_field)
    if root is None or not all((root / f).is_file() for f in ("ffnonbonded.itp", "ffbonded.itp")):
        raise ModuleConfigError(f"The complete {force_field} parameter database is required")
    masses = {
        fields[0]: float(fields[-5])
        for section, fields in _records(root / "ffnonbonded.itp")
        if section == "atomtypes"
    }
    parameters = {name: [] for name in ("bonds", "angles", "dihedrals", "impropers")}
    for section, f in _records(root / "ffbonded.itp"):
        if section == "bondtypes" and f[2] == "1":
            r0, k = map(float, f[3:5])
            parameters["bonds"].append((tuple(f[:2]), (k / 836.8, r0 * 10)))
        elif section == "angletypes" and f[3] in {"1", "5"}:
            theta, k = map(float, f[4:6])
            values = (k / 8.368, theta)
            if f[3] == "5":
                r13, kub = map(float, f[6:8])
                values += (kub / 836.8, r13 * 10)
            parameters["angles"].append((tuple(f[:3]), values))
        elif section == "dihedraltypes" and f[4] == "9":
            phase, k, multiplicity = map(float, f[5:8])
            parameters["dihedrals"].append((tuple(f[:4]), (k / 4.184, multiplicity, phase)))
        elif section == "dihedraltypes" and f[4] == "2":
            phase, k = map(float, f[5:7])
            parameters["impropers"].append((tuple(f[:4]), (k / 8.368, phase)))
    return masses, parameters
