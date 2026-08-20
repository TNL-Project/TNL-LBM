"""Unit tests compiled into the C++ unit-test binary (test_cpp_units).

Drives the compiled test_cpp_units executable — the single doctest binary that
aggregates every unit-test translation unit in tests/unit/ (outflow-pass
rectangle cover checks, lattice decomposition neighbor discovery, and
multi-rank tiling/reciprocity invariants).

Cases are grouped by .cu source file (doctest test suite) and rank count:
each single-rank suite runs in one invocation, and multi-rank cases are
grouped by their ``np<N>`` name tag (e.g. ``np2`` cases share one
``mpirun -np 2`` invocation).

doctest's ``--test-case=`` flag is last-flag-wins, so all selected cases are
comma-separated into a single flag.  The expected case count is verified
against the binary's ``--count`` output before each run so that a missed
selection is a hard failure, not a silent skip.
doctest exits nonzero if any case fails, printing the failing case name
and assertion in its output.
"""

from __future__ import annotations

import pathlib
import re
import subprocess

import pytest

from tests.lbmtest import BUILD_DIR, run_sim

TEST_BINARY = BUILD_DIR / "tests" / "unit" / "test_cpp_units"


def _binary_ok() -> tuple[bool, str]:
    """Check that TEST_BINARY exists and runnable (doctest --help works)."""
    try:
        proc = subprocess.run(
            [str(TEST_BINARY), "--help", "--no-colors"],
            capture_output=True,
            text=True,
            check=False,
            timeout=60.0,
        )
        return proc.returncode == 0, proc.stderr
    except (OSError, subprocess.TimeoutExpired) as exc:
        return False, str(exc)


def _list_suites() -> list[str]:
    """Query TEST_BINARY for its registered doctest test suite names."""
    proc = subprocess.run(
        [str(TEST_BINARY), "--list-test-suites", "--no-colors"],
        capture_output=True,
        text=True,
        check=False,
        timeout=60.0,
    )
    if proc.returncode != 0:
        return []
    suites = []
    for line in proc.stdout.splitlines():
        name = line.strip()
        if name and name != "=" * len(name) and not name.startswith("[doctest]"):
            suites.append(name)
    return suites


def _count_cases(extra_args: list[str]) -> int:
    """Return the number of test cases matching the given doctest arguments."""
    cmd = [str(TEST_BINARY), "--count", "--no-colors", *extra_args]
    proc = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        check=False,
        timeout=60.0,
    )
    if proc.returncode != 0:
        return -1
    m = re.search(r"passing the current filters:\s*(\d+)", proc.stdout)
    return int(m.group(1)) if m else -1


def _discover_multi_rank(suite_name: str) -> dict[int, list[str]]:
    """Find multi-rank test cases in a suite, keyed by np count."""
    proc = subprocess.run(
        [
            str(TEST_BINARY),
            "--list-test-cases",
            f"--test-suite={suite_name}",
            "--no-colors",
        ],
        capture_output=True,
        text=True,
        check=False,
        timeout=60.0,
    )
    if proc.returncode != 0:
        return {}
    groups: dict[int, list[str]] = {}
    in_list = False
    for line in proc.stdout.splitlines():
        stripped = line.strip()
        if stripped == "[doctest] listing all test case names":
            in_list = True
            continue
        if stripped.startswith("[doctest]") and in_list:
            break
        if not in_list or not stripped or stripped == "=" * len(stripped):
            continue
        if "multi-rank" not in stripped:
            continue
        m = re.search(r"\bnp(\d+)\b", stripped)
        np_ranks = int(m.group(1)) if m else 2
        groups.setdefault(np_ranks, []).append(stripped)
    return groups


def _discover_single_count(suite_name: str) -> int:
    """Count the non-multi-rank cases in a suite."""
    return _count_cases(
        [f"--test-suite={suite_name}", "--test-case-exclude=*multi-rank*"]
    )


_binary_available, _binary_error = _binary_ok()

if _binary_available:
    _suites = _list_suites()
else:
    _suites = []

# Each entry: (label, np_ranks, extra_args, expected_count)
# extra_args are appended verbatim to the doctest command line.
Batch = tuple[str, int, list[str], int]

_batches: list[Batch] = []
for _suite in _suites:
    # single-rank cases (exclude multi-rank)
    np1_count = _discover_single_count(_suite)
    if np1_count > 0:
        _batches.append(
            (
                _suite,
                1,
                [f"--test-suite={_suite}", "--test-case-exclude=*multi-rank*"],
                np1_count,
            )
        )
    # multi-rank cases
    for np_ranks, names in sorted(_discover_multi_rank(_suite).items()):
        _batches.append(
            (
                f"{_suite} np{np_ranks}",
                np_ranks,
                [f"--test-case={','.join(sorted(names))}"],
                len(names),
            )
        )

BATCHES: list[Batch] = _batches
BATCH_IDS = [label for label, _, _, _ in BATCHES]

# TDD lock suites of the Schönherr ch7 AMR conversion (plan row 3, commit 3
# of .omo/plans/schonherr-ch7-conversion.md): the registration/parity/
# conservation census and geometry fingerprint cases in
# tests/unit/test_amr_schonherr_registration.cu assert the POST-conversion
# band geometry from the ruling formulas. They were carried xfail(strict)
# through the declared red window (commits 4--6) via an XFAIL_SUITE_REASONS
# marks table on this wrapper; the marks were REMOVED at the commit-7
# stage-1 gate where the geometry landed and the suite passes outright.

BATCH_PARAMS = [pytest.param(batch, id=batch[0]) for batch in BATCHES]


@pytest.mark.parametrize("batch", BATCH_PARAMS)
def test_cpp_units(
    batch: Batch,
    test_dir: pathlib.Path,
) -> None:
    """Run a batch of doctest cases in one subprocess.

    Each batch is either a full test suite (single-rank, ``--test-suite=``)
    or a rank-count group (multi-rank, ``--test-case=`` with comma-separated
    names).  The expected case count is pre-verified via ``--count`` so that
    a stale filter silently selecting zero cases is always a hard failure.
    """
    label, np_ranks, extra_args, expected = batch
    if not _binary_available:
        pytest.fail(
            f"cannot run {TEST_BINARY} — build the target first: "
            f"cmake --build BUILD_DIR --target test_cpp_units\n{_binary_error}",
            pytrace=False,
        )
    if expected <= 0:
        pytest.fail(f"no test cases found for batch {label!r}")
    # verify selection before running
    verify_cmd = [str(TEST_BINARY), "--count", "--no-colors", *extra_args]
    proc = subprocess.run(
        verify_cmd, capture_output=True, text=True, check=False, timeout=60.0
    )
    if proc.returncode != 0:
        pytest.fail(f"filter check failed for {label!r}: {proc.stderr}")
    m = re.search(r"passing the current filters:\s*(\d+)", proc.stdout)
    actual = int(m.group(1)) if m else -1
    if actual != expected:
        pytest.fail(
            f"filter selects {actual} cases, expected {expected}: {extra_args!r}"
        )
    run_sim(
        [str(TEST_BINARY), *extra_args],
        workdir=test_dir,
        np_ranks=np_ranks,
        timeout=60.0,
    )
