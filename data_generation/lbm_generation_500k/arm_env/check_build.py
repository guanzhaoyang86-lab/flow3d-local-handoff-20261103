"""Check whether the recorded ARM build can be reused without rebuilding it."""
from __future__ import annotations

import argparse
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import platform
import re
import subprocess
import sys


ROOT = Path(__file__).resolve().parent
PROFILE = "deltaai-arm-llvm20"
TAICHI_COMMIT = "93c190f603b676ec1f8871e4f46cf5d5be89548e"
LLVM_VERSION = "20.1.8"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def source_metadata(root: Path = ROOT) -> dict:
    source = root / "taichi_source"
    commit = subprocess.check_output(
        ["git", "-C", str(source), "rev-parse", "HEAD"], text=True
    ).strip()
    if commit != TAICHI_COMMIT:
        raise ValueError("Taichi source is not at the pinned ARM commit")
    patch = (root / "compiler.patch").read_bytes()
    patch_hash = (root / "compiler.patch.sha256").read_text().split()[0]
    if not re.fullmatch(r"[0-9a-f]{64}", patch_hash):
        raise ValueError("Invalid compiler patch SHA256")
    if hashlib.sha256(patch).hexdigest() != patch_hash:
        raise ValueError("Saved compiler patch does not match its SHA256")
    # Compare tracked source changes, without treating generated, untracked
    # submodule files as compiler changes.
    actual_patch = subprocess.check_output([
        "git", "-C", str(source), "diff", "--binary", "--no-color", "--no-ext-diff",
        "--no-textconv", "--ignore-submodules=untracked", "HEAD", "--",
    ])
    if actual_patch != patch:
        raise ValueError("Actual Taichi source diff differs from compiler.patch")
    return {
        "taichi_source_commit": commit,
        "compiler_patch_sha256": patch_hash,
        "source_diff_sha256": hashlib.sha256(actual_patch).hexdigest(),
    }


def runtime_metadata() -> dict:
    import taichi as ti
    import torch

    commit = ti._lib.core.get_commit_hash()
    llvm = ti._lib.core.get_llvm_target_support()
    if (not isinstance(commit, str)
            or not re.fullmatch(r"[0-9a-f]{7,40}", commit)
            or not TAICHI_COMMIT.startswith(commit)):
        raise ValueError("Installed Taichi native commit is not the pinned ARM build")
    if llvm != LLVM_VERSION:
        raise ValueError(f"Installed Taichi must use LLVM {LLVM_VERSION}, got {llvm!r}")
    if platform.system() != "Linux" or platform.machine() != "aarch64":
        raise ValueError("The ARM build requires Linux aarch64")
    package_version = version("taichi")
    native_version = ti._lib.core.get_version_string()
    module_version = (
        ".".join(map(str, ti.__version__))
        if isinstance(ti.__version__, tuple) else str(ti.__version__)
    )
    distribution_version = re.fullmatch(
        r"(\d+\.\d+\.\d+)(?:(?:a|b|rc)\d+|\.dev\d+|\.post\d+)?(?:\+[A-Za-z0-9.]+)?",
        package_version,
    )
    if (module_version != native_version or distribution_version is None
            or distribution_version.group(1) != native_version):
        raise ValueError("Installed Taichi Python, native and package versions differ")
    return {
        "profile": PROFILE,
        "system": platform.system(), "machine": platform.machine(),
        "python_executable": sys.executable,
        "python_version": platform.python_version(),
        "taichi_version": package_version, "taichi_commit": commit,
        "llvm_version": llvm,
        "torch_version": torch.__version__, "cuda_runtime": torch.version.cuda,
    }


def check_build(root: Path = ROOT) -> dict:
    report = json.loads((root / "build_report.json").read_text())
    if (not isinstance(report, dict)
            or type(report.get("schema_version")) is not int
            or report["schema_version"] != 1
            or report.get("status") != "compiled_and_installed"):
        raise ValueError("No completed ARM build report")
    expected = {**source_metadata(root), **runtime_metadata()}
    # Older reports already recorded the same patch digest; verifying the
    # current diff against those exact patch bytes provides the same check.
    if "source_diff_sha256" not in report:
        expected.pop("source_diff_sha256")
    for key, value in expected.items():
        if report.get(key) != value:
            raise ValueError(f"Build report {key} does not match the current build/runtime")
    wheel = Path(report["wheel"])
    if not wheel.is_absolute():
        wheel = root / wheel
    if file_sha256(wheel) != report.get("wheel_sha256"):
        raise ValueError("Recorded wheel SHA256 does not match its file")
    return report


def reusable_validation_marker(root: Path, expected: dict, production_gate):
    """Return the existing valid marker without changing its original bytes."""
    marker = root / "validated_env.json"
    try:
        saved = json.loads(marker.read_text())
        if not isinstance(saved, dict):
            return None
        if any(saved.get(key) != value for key, value in expected.items()):
            return None
        production_gate(PROFILE, marker)
    except (OSError, ValueError, TypeError, KeyError):
        return None
    return saved


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-build", action="store_true", required=True)
    parser.parse_args()
    try:
        check_build()
    except (OSError, ValueError, TypeError, KeyError, IndexError, AttributeError,
            ImportError, subprocess.SubprocessError) as error:
        print(f"ARM build cannot be reused: {error}", file=sys.stderr)
        return 1
    print("Reusing the verified ARM Taichi build and its existing report")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
