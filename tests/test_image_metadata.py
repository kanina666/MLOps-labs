import importlib.util
from pathlib import Path

import pytest

_path = Path(__file__).parents[1] / "scripts" / "image_version.py"
_spec = importlib.util.spec_from_file_location("image_version", _path)
assert _spec is not None
assert _spec.loader is not None
_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_module)
image_metadata = _module.image_metadata

SHA = "a" * 40


def test_main_image_has_sha_and_moving_main_tag() -> None:
    result = image_metadata("Owner/MLOps-labs", "refs/heads/main", SHA, "0.1.0")
    assert result["app_version"] == "0.1.0+sha.aaaaaaaaaaaa"
    assert result["tags"].splitlines() == [
        f"ghcr.io/owner/mlops-labs:sha-{SHA}",
        "ghcr.io/owner/mlops-labs:main",
    ]


def test_release_version_matches_runtime_version() -> None:
    result = image_metadata("owner/repo", "refs/tags/v1.2.3", SHA, "1.2.3")
    assert result["app_version"] == "1.2.3"
    assert result["tags"] == "ghcr.io/owner/repo:1.2.3"


def test_pull_request_never_gets_main_tag() -> None:
    result = image_metadata("owner/repo", "refs/pull/12/merge", SHA, "0.1.0")
    assert result["tags"] == f"ghcr.io/owner/repo:sha-{SHA}"


@pytest.mark.parametrize(
    ("repository", "ref", "sha", "version"),
    [
        ("owner/repo\nforged", "refs/heads/main", SHA, "0.1.0"),
        ("owner/repo", "refs/heads/main", "invalid", "0.1.0"),
        ("owner/repo", "refs/heads/main", SHA, "invalid"),
        ("owner/repo", "refs/tags/v1.2.3", SHA, "0.1.0"),
        ("owner/repo", "refs/tags/v1.2.3-rc1", SHA, "1.2.3"),
    ],
)
def test_invalid_release_metadata_is_rejected(repository, ref, sha, version) -> None:
    with pytest.raises(ValueError, match="Invalid|Expected|project.version|Release"):
        image_metadata(repository, ref, sha, version)
