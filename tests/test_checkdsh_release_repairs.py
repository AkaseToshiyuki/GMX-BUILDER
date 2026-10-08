"""Release boundaries exercised in disposable repositories, never live remotes."""

import os
import subprocess
import sys
from pathlib import Path

import pytest
from click.testing import CliRunner

from gmxbuilder.app import main

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import check_public_export as guard  # noqa: E402
import prepare_public_release as release  # noqa: E402


@pytest.mark.parametrize(
    "url,allowed",
    [
        ("https://github.com/AkaseToshiyuki/GMXBUILDER.git", True),
        ("git@github.com:AkaseToshiyuki/GMXBUILDER.git", True),
        ("ssh://git@github.com/AkaseToshiyuki/GMXBUILDER.git", True),
        ("https://evil.example/AkaseToshiyuki/GMXBUILDER.git", False),
        ("https://github.com/AkaseToshiyuki/GMXBUILDER-copy.git", False),
        ("https://github.com@evil.example/AkaseToshiyuki/GMXBUILDER", False),
        ("https://github.com/AkaseToshiyuki/GMXBUILDER.git?redirect=evil", False),
    ],
)
def test_remote_identity(url, allowed):
    assert guard.is_private_remote(url) is allowed


@pytest.fixture
def repository(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", os.devnull)

    def git(*arguments):
        return subprocess.check_output(
            ["git", *arguments], text=True, stderr=subprocess.PIPE
        ).strip()

    git("init", "-q")
    git("config", "user.name", "Fixture")
    git("config", "user.email", "fixture@example.invalid")
    (tmp_path / "README").write_text("public")
    git("add", ".")
    git("commit", "-qm", "base")
    baseline = git("rev-parse", "HEAD")
    monkeypatch.setattr(guard, "APPROVED_PUBLIC_BASE", baseline)
    return tmp_path, git, baseline


def test_history_and_parent_reject_private_ancestor_after_deletion(repository):
    root, git, baseline = repository
    assert guard.validate_public_parent(baseline) == baseline
    (root / "Private").mkdir()
    secret = root / "Private/secret\t中文\n.txt"
    secret.write_text("private")
    git("add", ".")
    git("commit", "-qm", "private")
    private = git("rev-parse", "HEAD")
    assert guard.private_paths(guard.tracked_paths(private))
    # A new filtered orphan tree must omit the literal, not quoted, path.
    tree, published, withheld = release.build_public_tree(private)
    assert "Private/secret\t中文\n.txt" in withheld
    assert not guard.private_paths(guard.tracked_paths(tree))
    assert published == ["README"]
    secret.unlink()
    git("add", "-u")
    git("commit", "-qm", "remove")
    assert not guard.private_paths(guard.tracked_paths("HEAD"))
    assert guard.history_private_paths("HEAD")
    with pytest.raises(ValueError, match="private material"):
        guard.validate_public_parent("HEAD")
    git("tag", "-a", "candidate", "-m", "tag")
    assert guard.history_private_paths("candidate")
    # A replacement object cannot conceal the real ancestry from the guard.
    git("replace", private, baseline)
    assert guard.history_private_paths("HEAD")
    (root / ".git/info/grafts").write_text(f"{git('rev-parse', 'HEAD')} {baseline}\n")
    assert guard.history_private_paths("HEAD")


def test_public_export_preserves_literal_filename_bytes(repository):
    root, git, _baseline = repository
    names = [b"sample\r.txt", b"sample\n.txt", b"sample-\xff.txt"]
    for index, name in enumerate(names):
        with open(os.fsencode(root) + b"/" + name, "wb") as handle:
            handle.write(str(index).encode())
    git("add", ".")
    git("commit", "-qm", "literal public names")
    tree, _published, _withheld = release.build_public_tree("HEAD")
    original = subprocess.check_output(["git", "ls-tree", "-rz", "HEAD"])
    exported = subprocess.check_output(["git", "ls-tree", "-rz", tree])
    assert exported == original


def test_unrelated_parent_is_refused(repository):
    _root, git, _baseline = repository
    tree = git("rev-parse", "HEAD^{tree}")
    orphan = git("commit-tree", tree, "-m", "unrelated")
    assert not guard.history_private_paths(orphan)
    git("replace", orphan, _baseline)
    with pytest.raises(ValueError, match="approved public history"):
        guard.validate_public_parent(orphan)


@pytest.mark.parametrize("suffix", [".installing", ".backup"])
def test_legacy_asset_staging_is_private(suffix):
    for target in ("charmm36", "charmm36m", "amber14sb_ol24"):
        assert guard.classify(f"src/gmxbuilder/data/forcefields/{target}{suffix}/secret.rtp")


def test_build_refuses_existing_output_before_pipeline(tmp_path, monkeypatch):
    from gmxbuilder.pipeline.pipeline import Pipeline

    output = tmp_path / "result"
    (output / "structure").mkdir(parents=True)
    original = output / "structure/irreplaceable.txt"
    original.write_bytes(b"keep original")
    config = tmp_path / "build.yaml"
    config.write_text(f"output_dir: {output}\nmodules: {{}}\n")
    monkeypatch.setattr(
        Pipeline, "create_default", lambda: pytest.fail("Construction must not start")
    )
    result = CliRunner().invoke(main, ["build", "-c", str(config)])
    assert result.exit_code != 0
    assert "Output directory is not empty" in result.output
    assert original.read_bytes() == b"keep original"


def test_asset_merge_bootstraps_without_site_packages(tmp_path):
    installer = str(ROOT / "scripts/install_external_assets.py")
    temporary = str(tmp_path)
    script = (
        "import importlib.util, pathlib\n"
        f"s=importlib.util.spec_from_file_location('installer', {installer!r})\n"
        "m=importlib.util.module_from_spec(s);s.loader.exec_module(m)\n"
        "try:\n"
        f" m._merge_onto_base(pathlib.Path({temporary!r}), 'absent', "
        "pathlib.Path('unused'), {'name':'fixture'})\n"
        "except RuntimeError as e:\n"
        " assert 'merge base absent' in str(e), str(e)\n"
    )
    result = subprocess.run([sys.executable, "-S", "-c", script], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_built_distributions_exclude_installed_and_legacy_staging_assets(tmp_path):
    import tarfile
    import zipfile

    for name in ("pyproject.toml", "MANIFEST.in", "uv.lock", "README.md", "LICENSE"):
        (tmp_path / name).write_bytes((ROOT / name).read_bytes())
    package = tmp_path / "src/gmxbuilder"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("")
    for target in ("charmm36", "charmm36m", "amber14sb_ol24", "charmm36.installing"):
        marker = package / "data/forcefields" / target / "nested/private-marker.rtp"
        marker.parent.mkdir(parents=True)
        marker.write_text("private")
    public = package / "data/forcefields/amber99sb.ff/forcefield.itp"
    public.parent.mkdir(parents=True)
    public.write_text("public")
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import setuptools.build_meta as b; b.build_sdist('dist'); b.build_wheel('dist')",
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr[-4000:]
    with tarfile.open(next((tmp_path / "dist").glob("*.tar.gz"))) as archive:
        sdist = archive.getnames()
    with zipfile.ZipFile(next((tmp_path / "dist").glob("*.whl"))) as archive:
        wheel = archive.namelist()
    for names in (sdist, wheel):
        assert not any("private-marker" in name for name in names)
        assert any(name.endswith("amber99sb.ff/forcefield.itp") for name in names)
    assert any(name.endswith("/uv.lock") for name in sdist)
