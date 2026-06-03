import shutil
import subprocess
from pathlib import Path

import numpy as np
import tvm_ffi
from tvm_ffi import libinfo


TEST4DSP_DIR = Path(__file__).resolve().parents[1]
SRC_FILE = TEST4DSP_DIR / "src" / "total_test.c"
BUILD_DIR = TEST4DSP_DIR / "build"
SO_FILE = BUILD_DIR / "libtotal_test.so"


def _build_total_test_so() -> Path:
    """Build the local FFI demo shared library when it is missing or stale."""
    compiler = shutil.which("cc") or shutil.which("gcc")
    if compiler is None:
        raise RuntimeError("Cannot find a C compiler to build libtotal_test.so")

    BUILD_DIR.mkdir(parents=True, exist_ok=True)
    if SO_FILE.exists() and SO_FILE.stat().st_mtime >= SRC_FILE.stat().st_mtime:
        return SO_FILE

    include_flags = [f"-I{path}" for path in libinfo.include_paths()]
    cmd = [
        compiler,
        "-shared",
        "-fPIC",
        "-O2",
        *include_flags,
        str(SRC_FILE),
        "-o",
        str(SO_FILE),
    ]
    subprocess.run(cmd, check=True)
    return SO_FILE


def test_my_custom_add():
    """Load and run the local TVM FFI C demo op through the public module API."""
    so_path = _build_total_test_so()
    mod = tvm_ffi.load_module(str(so_path))

    x = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    y = np.empty_like(x)

    mod.add_one_c(x, y)

    np.testing.assert_allclose(y, x + 1.0)
    print("Test passed!")

if __name__ == "__main__":
    test_my_custom_add()
