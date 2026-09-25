"""Write image metadata to GitHub Actions outputs when run as a script."""

import os
import tomllib
from pathlib import Path

from image_version import image_metadata


def main() -> None:
    project = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))
    metadata = image_metadata(
        os.environ["GITHUB_REPOSITORY"],
        os.environ["GITHUB_REF"],
        os.environ["GITHUB_SHA"],
        project["project"]["version"],
    )
    with Path(os.environ["GITHUB_OUTPUT"]).open("a", encoding="utf-8") as output:
        for key, value in metadata.items():
            output.write(f"{key}<<MLOPS_OUTPUT\n{value}\nMLOPS_OUTPUT\n")
    print(f"Image: {metadata['primary_tag']}; application: {metadata['app_version']}")


main()
