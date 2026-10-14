"""Build the static site: copy web/ and zip the Python package for Pyodide."""

import shutil
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SITE = ROOT / "site"
PACKAGE = ROOT / "lab_in_the_loop"


def build_site() -> None:
    """Write the site to site/, ready for GitHub Pages or a local file server.

    The browser downloads the zip and unpacks it into Pyodide, so the same Python
    code runs in the tests and in the browser.
    """
    shutil.rmtree(SITE, ignore_errors=True)
    shutil.copytree(ROOT / "web", SITE)
    (SITE / "py").mkdir()
    with zipfile.ZipFile(
        SITE / "py" / "lab_in_the_loop.zip", "w", compression=zipfile.ZIP_DEFLATED
    ) as archive:
        for source in sorted(PACKAGE.glob("*.py")):
            archive.write(source, arcname=f"lab_in_the_loop/{source.name}")


if __name__ == "__main__":
    build_site()
