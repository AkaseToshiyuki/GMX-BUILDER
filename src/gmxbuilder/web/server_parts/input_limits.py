"""Complexity limits for untrusted molecular-structure uploads."""

from __future__ import annotations

import io
import logging
import os
from dataclasses import dataclass

logger = logging.getLogger("gmxbuilder.web")


class StructureInputLimitError(ValueError):
    """Raised before an upload exceeds its scientific resource budget."""


def _bounded_environment_integer(name: str, default: int, hard_maximum: int) -> int:
    """Read one bounded integer setting, saying so when the value is refused.

    An out-of-range request used to fall back to the default silently, so an
    operator asking for 128 against a maximum of 64 quietly got 32 -- neither
    what they asked for nor the limit. Clamping to the boundary honours the
    intent as far as the limit allows, and both refusals are logged, because a
    resource limit that differs from the configured one without saying so is
    the kind of thing found only while debugging something else.
    """
    raw = os.environ.get(name, str(default)).strip()
    try:
        value = int(raw)
    except ValueError:
        if raw:
            logger.warning("%s=%r is not an integer; using the default of %d", name, raw, default)
        return default
    if value < 1:
        logger.warning("%s=%d is below the minimum of 1; using 1", name, value)
        return 1
    if value > hard_maximum:
        logger.warning(
            "%s=%d exceeds the supported maximum of %d; using %d",
            name,
            value,
            hard_maximum,
            hard_maximum,
        )
        return hard_maximum
    return value


@dataclass(frozen=True)
class StructureInputLimits:
    """Independent byte and complexity budgets for one uploaded structure."""

    max_bytes: int
    max_atoms: int
    max_records: int
    max_tokens: int
    max_line_bytes: int

    @classmethod
    def from_environment(cls) -> StructureInputLimits:
        max_megabytes = _bounded_environment_integer("GMXBUILDER_MAX_UPLOAD_MB", 32, 64)
        return cls(
            max_bytes=max_megabytes * 1024 * 1024,
            max_atoms=_bounded_environment_integer(
                "GMXBUILDER_MAX_STRUCTURE_ATOMS", 250_000, 1_000_000
            ),
            max_records=_bounded_environment_integer(
                "GMXBUILDER_MAX_STRUCTURE_RECORDS", 750_000, 2_000_000
            ),
            max_tokens=_bounded_environment_integer(
                "GMXBUILDER_MAX_STRUCTURE_TOKENS", 6_000_000, 20_000_000
            ),
            max_line_bytes=_bounded_environment_integer(
                "GMXBUILDER_MAX_STRUCTURE_LINE_BYTES", 1_048_576, 4_194_304
            ),
        )


@dataclass(frozen=True)
class StructurePayloadStats:
    records: int
    atom_records: int
    tokens: int


def inspect_structure_payload(
    content: bytes,
    structure_format: str,
    limits: StructureInputLimits,
) -> StructurePayloadStats:
    """Reject pathological text before the full PDB/mmCIF parser runs.

    This scan retains only counters and one input line at a time.  The parser's
    observed atom count is checked again after parsing because legal mmCIF may
    represent loop rows in forms that are not reliably counted line-by-line.
    """
    if len(content) > limits.max_bytes:
        raise StructureInputLimitError(
            f"Structure file exceeds the {limits.max_bytes // 1024 // 1024} MB limit"
        )
    if b"\x00" in content:
        raise StructureInputLimitError("Structure file contains NUL bytes")

    records = 0
    atom_records = 0
    tokens = 0
    is_pdb = structure_format.lower() == "pdb"
    for raw_line in io.BytesIO(content):
        records += 1
        if records > limits.max_records:
            raise StructureInputLimitError(
                f"Structure contains more than {limits.max_records:,} records"
            )
        if len(raw_line) > limits.max_line_bytes:
            raise StructureInputLimitError(
                f"Structure contains a line longer than {limits.max_line_bytes:,} bytes"
            )
        tokens += len(raw_line.split())
        if tokens > limits.max_tokens:
            raise StructureInputLimitError(
                f"Structure contains more than {limits.max_tokens:,} text tokens"
            )
        stripped = raw_line.lstrip()
        if is_pdb:
            record_name = raw_line[:6].strip().upper()
            atom_record = record_name in {b"ATOM", b"HETATM"}
        else:
            atom_record = stripped.startswith((b"ATOM ", b"HETATM "))
        if atom_record:
            atom_records += 1
            if atom_records > limits.max_atoms:
                raise StructureInputLimitError(
                    f"Structure contains more than {limits.max_atoms:,} atom records"
                )

    return StructurePayloadStats(records, atom_records, tokens)


def enforce_parsed_atom_limit(atom_count: int, limits: StructureInputLimits) -> None:
    """Apply the authoritative atom limit to the parser result."""
    if atom_count > limits.max_atoms:
        raise StructureInputLimitError(
            f"Structure contains {atom_count:,} atoms; maximum is {limits.max_atoms:,}"
        )
