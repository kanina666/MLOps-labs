"""Execute EDA with this Python interpreter; optionally upload to Compose MLflow."""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run_eda(*, upload: bool = False, tracking_uri: str | None = None) -> tuple[Path, Path]:
    jupyter_data = ROOT / ".jupyter" / "data"
    kernel = jupyter_data / "kernels" / "m5-eda"
    kernel.mkdir(parents=True, exist_ok=True)
    (kernel / "kernel.json").write_text(
        json.dumps(
            {
                "argv": [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"],
                "display_name": "Python (M5 EDA)",
                "language": "python",
            }
        ),
        encoding="utf-8",
    )
    os.environ["JUPYTER_PATH"] = os.pathsep.join(
        [
            str(jupyter_data),
            str(Path(sys.prefix) / "share" / "jupyter"),
            str(Path(sys.base_prefix) / "share" / "jupyter"),
        ]
    )
    for name, directory in [
        ("JUPYTER_RUNTIME_DIR", ".jupyter/runtime"),
        ("IPYTHONDIR", ".ipython"),
        ("MPLCONFIGDIR", ".matplotlib"),
    ]:
        path = ROOT / directory
        path.mkdir(parents=True, exist_ok=True)
        os.environ[name] = str(path)

    import nbformat
    from nbclient import NotebookClient
    from nbconvert import HTMLExporter

    notebook = nbformat.read(ROOT / "notebooks" / "M5_EDA_MLOps.ipynb", as_version=4)
    reports = ROOT / "reports"
    reports.mkdir(exist_ok=True)
    executed = reports / "M5_EDA_MLOps.executed.ipynb"
    html_path = reports / "M5_EDA_MLOps.html"

    def progress(cell, cell_index, **_kwargs):
        if cell.cell_type == "code":
            print(f"Executing cell {cell_index + 1}/{len(notebook.cells)}", flush=True)

    client = NotebookClient(
        notebook,
        timeout=1200,
        kernel_name="m5-eda",
        resources={"metadata": {"path": str(ROOT)}},
        on_cell_start=progress,
    )
    try:
        client.execute()
    finally:
        nbformat.write(notebook, executed)
    html, _resources = HTMLExporter().from_notebook_node(notebook)
    html_path.write_text(html, encoding="utf-8")
    print(f"Executed notebook: {executed}")
    print(f"HTML: {html_path}")
    if upload:
        command = [
            sys.executable,
            str(ROOT / "scripts" / "log_eda.py"),
            "--report",
            str(executed),
            "--report",
            str(html_path),
        ]
        if tracking_uri:
            command.extend(["--tracking-uri", tracking_uri])
        subprocess.run(command, cwd=ROOT, check=True)
    return executed, html_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--log-mlflow", action="store_true", help="Upload to the Compose tracking server"
    )
    parser.add_argument("--tracking-uri", default=None)
    parser.add_argument(
        "--data-dir", type=Path, help="Folder containing manifest.json and raw/*.csv"
    )
    args = parser.parse_args()
    if args.data_dir:
        os.environ["M5_DATA_DIR"] = str(args.data_dir.resolve())
    run_eda(upload=args.log_mlflow, tracking_uri=args.tracking_uri)


main()
