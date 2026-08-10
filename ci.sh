#!/usr/bin/env bash
# Every check, in one command. Run it before committing; CI runs the same file,
# so "it passed locally" and "it passed in CI" mean the same thing.
#
#   ./ci.sh            everything, including the clean-install check
#   ./ci.sh --quick    skip the clean-install check (~40 s faster)
#
# The clean-install check matters because a suite that only ever runs against
# the source tree cannot prove the built wheel is complete, and this package
# ships its Hydra config tree inside the wheel.
set -euo pipefail

cd "$(dirname "$0")"

QUICK=0
[ "${1:-}" = "--quick" ] && QUICK=1

step() { printf '\n\033[1m== %s\033[0m\n' "$1"; }
fail() { printf '\n\033[31mFAILED: %s\033[0m\n' "$1" >&2; exit 1; }

step "Environment (uv sync --frozen)"
# --frozen installs exactly what uv.lock records and refuses to update it, so a
# lockfile that has drifted from pyproject.toml fails here rather than being
# silently rewritten.
uv lock --check || fail "uv.lock is out of date with pyproject.toml — run 'uv lock'"
uv sync --frozen || fail "uv sync"

step "Optional backend (DISF via PyIFT)"
# Not fatal: DISF is an optional extra and the suite skips its tests without
# it. But when it is installed and cannot load, the cause is almost always a
# missing system library, and that is worth saying in one line rather than
# leaving as an ImportError inside test collection.
if ! uv run --frozen python -c "import pyift.pyift" 2>/dev/null; then
    printf '\033[33mPyIFT is not available; DISF tests will skip.\033[0m\n'
    printf 'If you meant to have it:  sudo apt-get install -y liblapack3 libblas3\n'
else
    printf 'PyIFT loads; DISF tests will run.\n'
fi

step "Lint (ruff check)"
uv run --frozen ruff check src tests examples || fail "ruff check"

step "Format (ruff format --check)"
uv run --frozen ruff format --check src tests examples \
    || fail "ruff format — run 'uv run ruff format src tests examples'"

step "Types (mypy --strict)"
uv run --frozen mypy src || fail "mypy"

step "Tests (pytest)"
uv run --frozen pytest -q || fail "pytest"

if [ "$QUICK" = "1" ]; then
    printf '\n\033[33mSkipped the clean-install check (--quick).\033[0m\n'
    printf '\033[32mAll checks passed.\033[0m\n'
    exit 0
fi

step "Clean install (wheel, no source tree on sys.path)"
BUILD_DIR="$(mktemp -d)"
trap 'rm -rf "$BUILD_DIR"' EXIT

uv build --wheel --out-dir "$BUILD_DIR/dist" >/dev/null || fail "uv build"
uv venv --python 3.12 "$BUILD_DIR/venv" >/dev/null || fail "uv venv"
PY="$BUILD_DIR/venv/bin/python"

# Install the built wheel — no editable install, no path back to src/, which is
# the entire point of this step.
uv pip install --python "$PY" --quiet "$BUILD_DIR"/dist/*.whl \
    || fail "installing the wheel"

# The console script must start. It reads its config tree from inside the
# installed package, which is exactly what a source checkout cannot verify.
"$BUILD_DIR/venv/bin/spifil-fit" --help >/dev/null \
    || fail "spifil-fit is broken when installed"

# And the package must be importable with nothing else present.
"$PY" -c "
import spifil
from spifil import Learner, SLIC, SpifilDataset
assert spifil.__version__
" || fail "the installed wheel does not import"

printf '\n\033[32mAll checks passed.\033[0m\n'
