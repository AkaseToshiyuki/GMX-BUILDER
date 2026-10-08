"""User force-field spellings cannot create paths or unbounded parser copies."""

import asyncio
from concurrent.futures import ThreadPoolExecutor

import pytest

from gmxbuilder.modules.forcefield import rtp_parser


@pytest.mark.parametrize(
    "name",
    ["amber14sb.ff/.", "../amber14sb.ff", "/tmp/amber14sb", "amber14sb\\x", "%2e%2e", "a" * 10000],
)
def test_non_catalog_spellings_rejected_before_loading(name, monkeypatch):
    monkeypatch.setattr(rtp_parser, "_rtp_by_force_field", {})
    with pytest.raises(FileNotFoundError, match="force field directory"):
        rtp_parser.load_force_field_rtp(name)
    assert not rtp_parser._rtp_by_force_field
    from gmxbuilder.web import server

    assert asyncio.run(server.api_terminal_capabilities(name)).status_code == 400


def test_canonical_aliases_and_concurrent_misses_share_one_parser(tmp_path, monkeypatch):
    (tmp_path / "aminoacids.rtp").write_text("[ ALA ]\n[ atoms ]\nCA CT 0.0 1\n")
    monkeypatch.setattr(rtp_parser, "_rtp_by_force_field", {})
    monkeypatch.setattr(rtp_parser, "_force_field_path", lambda name: tmp_path)
    calls = []
    original = rtp_parser.RTPParser.parse

    def parse(self, path):
        calls.append(path)
        original(self, path)

    monkeypatch.setattr(rtp_parser.RTPParser, "parse", parse)
    names = ["amber14sb", " AMBER14SB ", "amber14sb.ff"] * 8
    with ThreadPoolExecutor(max_workers=8) as pool:
        parsers = list(pool.map(rtp_parser.load_force_field_rtp, names))
    assert len(calls) == 1
    assert len({id(p) for p in parsers}) == 1
    assert parsers[0].get_atom_type("ALA", "CA") == ("CT", 0.0)
    assert set(rtp_parser._rtp_by_force_field) == {"amber14sb"}
