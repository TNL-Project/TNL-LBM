"""Unit tests compiled into the C++ unit-test binary (test_cpp_units).

Drives the compiled test_cpp_units executable — the single doctest binary that
aggregates every unit-test translation unit in tests/unit/ (currently the
greedy rectangle cover checks for the outflow-pass sites that
LBM_BLOCK<CONFIG>::updateOutflowPassRegion() computes from the host map).
The doctest test case names are discovered from the binary itself at
collection time (`--list-test-cases`); each case is one pytest item for
granular failure reports, so the items can never go stale.
"""

from __future__ import annotations

import pathlib
import subprocess

import pytest

from tests.lbmtest import BUILD_DIR, run_sim

TEST_BINARY = BUILD_DIR / "tests" / "unit" / "test_cpp_units"


def _discover_cases() -> tuple[list[str], str]:
    """Query TEST_BINARY for its registered doctest test case names.

    The verdict is the binary's exit code. The `--list-test-cases` output
    has a `[doctest] listing all test case names` header line, a separator
    line of `=` characters, one bare name per line, the separator again,
    and a `[doctest]`-prefixed count summary. Only lines between the header
    and the summary are considered — the MPI initialization noise printed
    to stdout before the header would otherwise be misparsed as names.

    Returns the case names, or an empty list and the reason when the binary
    cannot be queried (e.g. the build tree lacks the target) — the
    parametrized test then collapses to one failure item at run time
    instead of crashing the collection.
    """
    try:
        proc = subprocess.run(
            [str(TEST_BINARY), "--list-test-cases", "--no-colors"],
            capture_output=True,
            text=True,
            check=False,
            timeout=60.0,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return [], str(exc)
    if proc.returncode != 0:
        reason = f"{TEST_BINARY} --list-test-cases exited {proc.returncode}:\n"
        return [], reason + proc.stdout + proc.stderr
    lines = proc.stdout.splitlines()
    try:
        start = lines.index("[doctest] listing all test case names")
    except ValueError:
        return [], f"no listing header in `{TEST_BINARY} --list-test-cases` output"
    cases = []
    for line in lines[start + 1 :]:
        name = line.strip()
        if not name or set(name) == {"="}:
            continue  # separator or blank line
        if name.startswith("[doctest]"):
            break  # the trailing count summary terminates the listing
        cases.append(name)
    if not cases:
        return [], f"no test case names in `{TEST_BINARY} --list-test-cases` output"
    return cases, ""


CASES, LISTING_ERROR = _discover_cases()


@pytest.mark.parametrize("case", CASES or [None])
def test_cpp_units(case: str | None, test_dir: pathlib.Path) -> None:
    """Run one unit-test case; doctest exits nonzero on any failed check."""
    if case is None:
        pytest.fail(
            f"cannot list test cases from {TEST_BINARY} — build the target "
            "first: cmake --build BUILD_DIR --target test_cpp_units\n" + LISTING_ERROR,
            pytrace=False,
        )
    run_sim([TEST_BINARY, f"--test-case={case}"], workdir=test_dir, timeout=60.0)
