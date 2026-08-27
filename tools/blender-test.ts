#!/usr/bin/env bun
import { existsSync, mkdirSync } from "node:fs";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";
import { $ } from "bun";

const REPO_ROOT = resolve(import.meta.dir, "..");
const RUNNER = join(REPO_ROOT, "tests", "blender", "runner.py");
const PYLIBS = join(tmpdir(), "mtree-blender-pylibs");

const CANDIDATES = [
  process.env.BLENDER,
  "S:/SteamLibrary/steamapps/common/Blender/blender.exe",
  "C:/Program Files/Blender Foundation/Blender 5.2/blender.exe",
  "C:/Program Files/Blender Foundation/Blender 4.5/blender.exe",
  "/Applications/Blender.app/Contents/MacOS/Blender",
  "/usr/bin/blender",
].filter((path): path is string => Boolean(path));

function findBlender(): string {
  const found = CANDIDATES.find((path) => existsSync(path));
  if (!found) {
    console.error(
      "Blender not found. Set BLENDER=/path/to/blender.\nTried:\n  " +
        CANDIDATES.join("\n  "),
    );
    process.exit(2);
  }
  return found;
}

async function blenderPythonVersion(blender: string): Promise<string> {
  const probe = "import sys; print('MTREE_PY=%d.%d' % sys.version_info[:2])";
  const out =
    await $`${blender} -b --factory-startup --python-expr ${probe}`.text();
  const match = out.match(/MTREE_PY=(\d+\.\d+)/);
  if (!match) {
    console.error("Could not determine Blender's Python version.");
    process.exit(2);
  }
  return match[1];
}

async function ensurePytest(pythonVersion: string): Promise<void> {
  if (existsSync(join(PYLIBS, "pytest"))) return;

  mkdirSync(PYLIBS, { recursive: true });
  console.log(`Vendoring pytest for Python ${pythonVersion} into ${PYLIBS}`);

  await $`uv pip install --target ${PYLIBS} --python-version ${pythonVersion} --only-binary=:all: pytest`.quiet();
}

const blender = findBlender();
const pythonVersion = await blenderPythonVersion(blender);
await ensurePytest(pythonVersion);

const result = await $`${blender} -b --factory-startup --python ${RUNNER}`
  .env({
    ...process.env,
    MTREE_PYLIBS: PYLIBS,
    MTREE_PYTEST_ARGS: process.argv.slice(2).join(" "),
  })
  .nothrow();

process.exit(result.exitCode);
