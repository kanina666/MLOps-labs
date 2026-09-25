"""Calculate image tags without reading the environment or writing files."""

import re


def image_metadata(repository: str, ref: str, sha: str, package_version: str) -> dict[str, str]:
    if not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", repository):
        raise ValueError("Invalid GitHub repository name")
    if not re.fullmatch(r"[0-9a-f]{40}", sha):
        raise ValueError("Expected a full lowercase Git commit SHA")
    if not re.fullmatch(r"\d+\.\d+\.\d+", package_version):
        raise ValueError("project.version must use X.Y.Z for this lab's release policy")

    image = f"ghcr.io/{repository.lower()}"
    if ref.startswith("refs/tags/"):
        tag = ref.removeprefix("refs/tags/")
        if not re.fullmatch(r"v\d+\.\d+\.\d+", tag):
            raise ValueError("Release tags must use vX.Y.Z")
        app_version = tag.removeprefix("v")
        if app_version != package_version:
            raise ValueError("Release tag must match project.version in pyproject.toml")
        tags = [f"{image}:{app_version}"]
    else:
        app_version = f"{package_version}+sha.{sha[:12]}"
        tags = [f"{image}:sha-{sha}"]
        if ref == "refs/heads/main":
            tags.append(f"{image}:main")
    return {"app_version": app_version, "primary_tag": tags[0], "tags": "\n".join(tags)}
