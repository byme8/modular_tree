"""Entry point for the Blender test tier. Not a test module itself.

    blender.exe -b --factory-startup --python tests/blender/runner.py

Run through `bun tools/blender-test.ts`, which locates Blender, vendors pytest
into a temp directory and passes it here via MTREE_PYLIBS.

The repo root goes on sys.path ahead of everything else because the addon is
usually also installed as a Blender extension: without that precedence the tests
import the installed copy and silently verify stale code.
"""

import os
import sys
from pathlib import Path

TESTS_DIR = Path(__file__).resolve().parent
REPO_ROOT = TESTS_DIR.parent.parent

sys.path.insert(0, str(REPO_ROOT))

pylibs = os.environ.get("MTREE_PYLIBS")
if pylibs:
    sys.path.insert(0, pylibs)

try:
    import pytest
except ImportError:
    sys.stderr.write(
        "pytest not found. Set MTREE_PYLIBS to a directory containing it, or run "
        "via `bun tools/blender-test.ts`.\n"
    )
    sys.exit(2)


def main() -> int:
    args = ["-q", "--no-header", "-p", "no:cacheprovider", str(TESTS_DIR)]

    extra = os.environ.get("MTREE_PYTEST_ARGS")
    if extra:
        args.extend(extra.split())

    return int(pytest.main(args))


if __name__ == "__main__":
    sys.exit(main())
