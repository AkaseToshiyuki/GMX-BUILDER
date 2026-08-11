"""Canonical all-atom nucleic-acid support for solution systems."""

from .support import (
    CANONICAL_DNA_RESNAMES,
    CANONICAL_RNA_RESNAMES,
    classify_nucleic_residue,
    is_nucleic_like_residue,
)

__all__ = [
    "CANONICAL_DNA_RESNAMES",
    "CANONICAL_RNA_RESNAMES",
    "classify_nucleic_residue",
    "is_nucleic_like_residue",
]
