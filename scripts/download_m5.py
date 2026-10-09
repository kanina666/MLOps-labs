"""Fetch the pinned M5 mirror and verify the committed data/manifest.json hashes."""

import hashlib
import json
import shutil
import zipfile
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main() -> None:
    raw = ROOT / "data" / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((ROOT / "data" / "manifest.json").read_text(encoding="utf-8"))
    for record in manifest["files"]:
        target = raw / record["filename"]
        if target.exists():
            if digest(target) != record["sha256"]:
                raise ValueError(f"Existing data has an unexpected hash: {target}")
            print(f"Verified: {target.name}")
            continue
        download = raw / (target.name + ".download")
        with requests.get(record["source_url"], stream=True, timeout=(20, 180)) as response:
            response.raise_for_status()
            with download.open("wb") as stream:
                for chunk in response.iter_content(1024 * 1024):
                    stream.write(chunk)
        part = raw / (target.name + ".part")
        if record["source_url"].endswith(".zip"):
            with zipfile.ZipFile(download) as archive:
                members = [name for name in archive.namelist() if Path(name).name == target.name]
                if len(members) != 1:
                    raise ValueError(f"Unexpected archive contents for {target.name}")
                with archive.open(members[0]) as source, part.open("wb") as stream:
                    shutil.copyfileobj(source, stream)
        else:
            shutil.copyfile(download, part)
        if digest(part) != record["sha256"]:
            raise ValueError(f"Downloaded bytes do not match the pinned hash: {target.name}")
        part.replace(target)
        download.unlink()
        print(f"Downloaded and verified: {target.name}")


if __name__ == "__main__":
    main()
