"""Brand asset and public-document integration checks."""

from pathlib import Path
import struct


ROOT = Path(__file__).resolve().parents[1]
LOGO_RELATIVE = "src/gmxbuilder/web/static/assets/gmxbuilder-logo.png"
LOGO = ROOT / LOGO_RELATIVE


def test_logo_is_transparent_high_resolution_png() -> None:
    data = LOGO.read_bytes()
    assert data.startswith(b"\x89PNG\r\n\x1a\n")
    width, height, _bit_depth, color_type = struct.unpack(">IIBB", data[16:26])
    assert width >= 1000
    assert height >= 300
    assert color_type in {4, 6}, "the Web logo must retain an alpha channel"


def test_web_header_uses_versioned_accessible_logo() -> None:
    template = (ROOT / "src/gmxbuilder/web/templates/index.html").read_text()
    assert 'class="brand-logo"' in template
    assert 'src="/static/assets/gmxbuilder-logo.png?v={{ version }}"' in template
    assert 'alt="GMXBUILDER"' in template
    assert 'href="/static/assets/gmxbuilder-mark.png?v={{ version }}"' in template
    assert 'class="release-badge">Beta Test v{{ version }}' in template
    assert "Alpha Test" not in template


def test_both_readmes_use_the_canonical_logo_asset() -> None:
    for name in ("README.md", "README.zh-CN.md"):
        readme = (ROOT / name).read_text()
        assert LOGO_RELATIVE in readme


def test_public_documentation_points_to_current_user_manual() -> None:
    manual_stem = "GMXBUILDER_USER_MANUAL_V1.0.3"
    for suffix in (".md", ".pdf", ".zh-CN.md", ".zh-CN.pdf"):
        assert (ROOT / "docs" / f"{manual_stem}{suffix}").is_file()
    for name in ("README.md", "README.zh-CN.md", "docs/README.md", "docs/README.zh-CN.md"):
        content = (ROOT / name).read_text()
        assert manual_stem in content
        assert "GMXBUILDER_USER_MANUAL_V1.0.0" not in content

    source = (ROOT / "docs" / f"{manual_stem}.md").read_text()
    chinese = (ROOT / "docs" / f"{manual_stem}.zh-CN.md").read_text()
    assert "| Document version | V1.0.3 |" in source
    assert "| Software | GMXBUILDER v0.9.9 or later |" in source
    assert "| V1.0.3 | 2026-08-15 |" in source
    assert "| V1.0.0 | 2026-07-26 | Initial release |" in source
    assert "| 文档版本 | V1.0.3 |" in chinese
    assert "| 适用软件 | GMXBUILDER v0.9.9 或更高版本 |" in chinese
    assert "## 附录 B：文档维护要求" not in chinese


def test_public_landing_pages_are_bilingual_without_internal_roadmaps() -> None:
    english = (ROOT / "README.md").read_text()
    chinese = (ROOT / "README.zh-CN.md").read_text()
    assert '<strong>English</strong> · <a href="README.zh-CN.md">简体中文</a>' in english
    assert '<a href="README.md">English</a> · <strong>简体中文</strong>' in chinese
    assert "```mermaid" not in english and "## Architecture" not in english
    assert "```mermaid" not in chinese and "## 架构" not in chinese
    assert "access token" in english
    assert "不要求 GitHub Token" in " ".join(chinese.split())
    for readme in (english, chinese):
        assert "artifact" not in readme.lower()
        assert "制品" not in readme
    for name in (
        "WORKFLOW_SPECIFICATION.md",
        "NUCLEIC_ACID_AND_LIGAND_SUPPORT.md",
        "AUDIT_AND_ROADMAP.md",
        "COARSE_GRAINED_MARTINI3_ROADMAP.md",
        "GITHUB_RELEASE_CHECKLIST.md",
        "TEST_ACCEPTANCE.md",
    ):
        assert not (ROOT / "docs" / name).exists()


def test_each_public_user_document_has_a_language_switch() -> None:
    pairs = (
        ("README.md", "README.zh-CN.md"),
        ("docs/README.md", "docs/README.zh-CN.md"),
        (
            "docs/GMXBUILDER_USER_MANUAL_V1.0.3.md",
            "docs/GMXBUILDER_USER_MANUAL_V1.0.3.zh-CN.md",
        ),
        (
            "docs/SCIENTIFIC_COMPATIBILITY.md",
            "docs/SCIENTIFIC_COMPATIBILITY.zh-CN.md",
        ),
        ("THIRD_PARTY_NOTICES.md", "THIRD_PARTY_NOTICES.zh-CN.md"),
    )
    for english_name, chinese_name in pairs:
        english = (ROOT / english_name).read_text()
        chinese = (ROOT / chinese_name).read_text()
        assert Path(chinese_name).name in english
        assert Path(english_name).name in chinese


def test_frontend_header_uses_only_the_selected_workflow_name() -> None:
    template = (ROOT / "src/gmxbuilder/web/templates/index.html").read_text()
    script = (ROOT / "src/gmxbuilder/web/static/app.js").read_text()
    assert 'id="header-task-title"></span>' in template
    assert "GROMACS Molecular System Builder" not in template
    assert "GROMACS Molecular System Builder" not in script
    assert "state.taskType.title;" in script
    assert "state.taskType.description.substring" not in script


def test_completed_build_resume_renders_existing_result() -> None:
    script = (ROOT / "src/gmxbuilder/web/static/app.js").read_text()
    ions = (ROOT / "src/gmxbuilder/web/static/ions.js").read_text()
    assert 'resumedBuild.status === "completed"' in script
    assert "_showBuildResult(resumedBuild.result)" in script
    assert "window._setSystemConfirmed(canRestoreSystemConfirmation)" in script
    assert "window._setSystemConfirmed = function(v)" in ions


def test_original_user_supplied_artwork_is_preserved() -> None:
    original = LOGO.with_name("gmxbuilder-logo-original.jpg")
    assert original.is_file()
    assert original.stat().st_size > 10_000
