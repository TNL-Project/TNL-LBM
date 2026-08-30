"""AMR bit-identity evidence harness (plan amr-nlevel-nesting §7.5).

Instrument for the A-E commits of the multi-level nesting plan: proves that a
commit changed NO runtime behavior of the existing single-fine-level
configuration by re-running the reference battery in the current build tree
and comparing every artifact byte-for-byte against the reference digests
committed in ``tests/regression/amr_ref/manifest.json``.

Battery (all single rank, current-tree binaries under ``TNL_LBM_BUILD_DIR``):

- the mock census doctest suites ``amr_coupling``, ``amr_subcycling`` and
  ``amr_vtkhdf_writer`` of the per-pattern consolidated binaries
  ``test_amr_units_{ab,aa}``, each run via doctest's ``--test-suite``
  filter (the artifact keys keep the historical per-suite names
  ``test_amr_{coupling,subcycling,vtkhdf_writer}_{ab,aa}`` so the manifest
  stays comparable) — their complete stdout (normalized, see below) plus
  the dataset content of every ``*.vtkhdf`` file they emit
  (``amr_vtkhdf_writer`` writes ``test_amr.vtkhdf``);
- short fixed-configuration runs of ``sim_AMR --resolution 1`` and
  ``sim_AMR_channel --resolution 1`` (both default settings) — normalized
  stdout (carries the conservation lines) plus the dataset content of every
  ``results_*/output_amr_*.vtkhdf`` frame.

stdout normalization removes only provably volatile content: UCX interface
probes, leading ``[YYYY-MM-DD ...]`` timestamps, the ``GLUPS=`` performance
lines, the ``total walltime:`` line, and the memory-availability totals
(system-RAM dependent). Every remaining byte is part of the identity claim.

VTKHDF files are compared by DATASET content (dataset names, shapes, dtypes,
raw array bytes and HDF5 attributes), not by file bytes: the ADIOS2 HDF5
container embeds volatile creation metadata, while the stored arrays are
bit-reproducible across identical runs on the same machine.

Final conservation values are additionally pinned to the literal acceptance
numbers of the plan gate, so a physics drift on ANY level fails loudly even
where a byte-diff would need a human to read it.

Modes:

- verify (default): compare the freshly recorded digests against
  ``amr_ref/manifest.json``; any digest or artifact-set difference fails.
- record (``TNL_LBM_AMR_REF=record``): re-record the battery and OVERWRITE
  the manifest; the comparing tests are skipped. Record only from a trusted
  tree state (the pre-change baseline), then commit the manifest.
"""

from __future__ import annotations

import hashlib
import json
import os
import pathlib
import re
import shutil

import pytest

from tests.lbmtest import ADIOS_CONFIG, BUILD_DIR, PROJECT_ROOT, run_sim

h5py = pytest.importorskip("h5py")

REF_DIR = PROJECT_ROOT / "tests" / "regression" / "amr_ref"
MANIFEST = REF_DIR / "manifest.json"

_RECORD = os.environ.get("TNL_LBM_AMR_REF") == "record"

# artifact key -> (consolidated doctest binary, TEST_SUITE filter inside it)
MOCK_SUITES: dict[str, tuple[str, str]] = {
    "test_amr_coupling_ab": ("test_amr_units_ab", "amr_coupling"),
    "test_amr_coupling_aa": ("test_amr_units_aa", "amr_coupling"),
    "test_amr_subcycling_ab": ("test_amr_units_ab", "amr_subcycling"),
    "test_amr_subcycling_aa": ("test_amr_units_aa", "amr_subcycling"),
    "test_amr_vtkhdf_writer_ab": ("test_amr_units_ab", "amr_vtkhdf_writer"),
    "test_amr_vtkhdf_writer_aa": ("test_amr_units_aa", "amr_vtkhdf_writer"),
}

SIMS: dict[str, tuple[pathlib.Path, list[str]]] = {
    "sim_AMR": (BUILD_DIR / "sim_AMR" / "sim_AMR", ["--resolution", "1"]),
    "sim_AMR_channel": (
        BUILD_DIR / "sim_AMR" / "sim_AMR_channel",
        ["--resolution", "1"],
    ),
}

# final conservation values pinned by the plan gate (printed with {:.6e})
PINNED_METRICS: dict[str, list[tuple[str, str]]] = {
    "sim_AMR": [
        ("AMR conservation: mass", "2.649349e+05"),
        ("AMR level 0: kinetic energy", "1.751046e+00"),
        ("AMR level 1: kinetic energy", "1.832126e+00"),
    ],
    "sim_AMR_channel": [
        ("AMR conservation: mass", "1.866006e+04"),
        ("AMR level 0: kinetic energy", "8.411523e+01"),
        ("AMR level 1: kinetic energy", "5.097540e+01"),
    ],
}

_ISO_TS = re.compile(r"^\[\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d{3}\] ")
_UCX_TS = re.compile(r"^\[\d+\.\d+\]")
# OpenMP-atomics association noise: floats below 1e-13 in the test logs are
# reduction-order epsilons, and the same reduction prints exactly-zero values
# as `0.000000e+00` or as a tiny epsilon run-to-run (the conservation
# assertions themselves carry explicit tolerances far above either), so
# neither class can be bit-stable
_FP_NOISE = re.compile(r"[+-]?\d\.\d+e-(?:1[3-9]|[2-9]\d)|[+-]?0\.0+e\+00")


def _normalize_stdout(text: str) -> str:
    """Strip provably volatile lines/tokens; keep the physics byte stream."""
    lines = []
    for line in text.splitlines():
        if _UCX_TS.match(line):
            continue  # UCX interface probe (timestamps, PIDs, host paths)
        if "GLUPS" in line:
            continue  # performance cadence lines (GLUPS/WT/ETA values)
        if "total walltime:" in line:
            continue
        if "compute time:" in line:
            continue  # phase-duration report, wall-clock volatile like the total
        if "saved in:" in line:
            continue  # write3D/write3Dcut wall-clock report lines
        if "MiB estimated needed," in line:
            continue  # available-RAM totals are system-load dependent
        stripped = _FP_NOISE.sub("<eps>", _ISO_TS.sub("", line))
        lines.append(stripped.rstrip())
    return "\n".join(lines) + "\n"


def _digest(text: str) -> str:
    return hashlib.md5(text.encode()).hexdigest()


def _vtkhdf_digest(path: pathlib.Path) -> str:
    """md5 over the HDF5 dataset content (names, shapes, dtypes, bytes, attrs)."""
    h = hashlib.md5()
    with h5py.File(path) as f:

        def visit(name: str, obj: h5py.Dataset | h5py.Group) -> None:
            h.update(name.encode())
            for key, value in sorted(obj.attrs.items()):
                h.update(key.encode())
                h.update(repr(value).encode())
            if isinstance(obj, h5py.Dataset):
                h.update(repr(obj.shape).encode())
                h.update(repr(obj.dtype).encode())
                h.update(obj[()].tobytes())

        for key, value in sorted(f.attrs.items()):
            h.update(key.encode())
            h.update(repr(value).encode())
        f.visititems(visit)
    return h.hexdigest()


def _collect_stdouts(
    root: pathlib.Path, conservation: dict[str, str]
) -> dict[str, str]:
    """Run the battery once and return {artifact: md5}; fill conservation."""
    artifacts: dict[str, str] = {}

    for suite, (binary_name, suite_filter) in MOCK_SUITES.items():
        binary = BUILD_DIR / "tests" / binary_name
        if not binary.is_file():
            pytest.fail(
                f"cannot find {binary} — build the AMR test targets first: "
                f"cmake --build {BUILD_DIR} --target {binary_name}",
                pytrace=False,
            )
        workdir = root / suite
        workdir.mkdir()
        # the suites read adios2.xml from their cwd (State ctor)
        shutil.copy(ADIOS_CONFIG, workdir / "adios2.xml")
        stdout = run_sim(
            [binary, f"--test-suite={suite_filter}", "--no-colors", "--no-duration"],
            workdir=workdir,
            timeout=300.0,
        )
        artifacts[f"{suite}.stdout"] = _digest(_normalize_stdout(stdout))
        for vtkhdf in sorted(workdir.glob("*.vtkhdf")):
            artifacts[f"{suite}.{vtkhdf.name}"] = _vtkhdf_digest(vtkhdf)

    for sim, (binary, args) in SIMS.items():
        if not binary.is_file():
            pytest.fail(
                f"cannot find {binary} — build the project first: "
                f"cmake --build {BUILD_DIR}",
                pytrace=False,
            )
        workdir = root / sim
        workdir.mkdir()
        stdout = run_sim(
            [binary, *args, "--adios-config", ADIOS_CONFIG],
            workdir=workdir,
            timeout=600.0,
        )
        artifacts[f"{sim}.stdout"] = _digest(_normalize_stdout(stdout))
        conservation[sim] = stdout
        frames = sorted(workdir.glob("results_*/output_amr_*.vtkhdf"))
        if not frames:
            pytest.fail(f"{sim} produced no results_*/output_amr_*.vtkhdf frames")
        for frame in frames:
            artifacts[f"{sim}.{frame.name}"] = _vtkhdf_digest(frame)

    return artifacts


class Battery:
    """Result of one battery run: artifact digests and raw sim stdout."""

    def __init__(self) -> None:
        self.artifacts: dict[str, str] = {}
        self.conservation: dict[str, str] = {}


def _battery(root: pathlib.Path) -> Battery:
    battery = Battery()
    battery.artifacts = _collect_stdouts(root, battery.conservation)
    if _RECORD:
        REF_DIR.mkdir(parents=True, exist_ok=True)
        manifest = {
            "schema": 1,
            "description": (
                "AMR bit-identity reference digests (plan amr-nlevel-nesting "
                "§7.5): md5 of normalized stdout and of VTKHDF dataset content "
                "for the mock census suites and sim_AMR / sim_AMR_channel "
                "short runs; recorded at the pre-refactor baseline and verified "
                "by tests/regression/test_amr_bitidentity.py"
            ),
            "artifacts": dict(sorted(battery.artifacts.items())),
        }
        MANIFEST.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return battery


@pytest.fixture(scope="module")
def battery(workspace: pathlib.Path) -> Battery:
    root = workspace / "amr_bitidentity"
    root.mkdir()
    return _battery(root)


def _reference_manifest() -> dict:
    if not MANIFEST.is_file():
        pytest.fail(
            f"reference manifest {MANIFEST} not found — record it first with "
            f"TNL_LBM_AMR_REF=record pytest tests/regression/test_amr_bitidentity.py",
            pytrace=False,
        )
    return json.loads(MANIFEST.read_text())


def _compare_keys(artifacts: dict[str, str], prefix: str) -> None:
    expected_all = _reference_manifest().get("artifacts", {})
    expected = {k: v for k, v in expected_all.items() if k.startswith(prefix)}
    actual = {k: v for k, v in artifacts.items() if k.startswith(prefix)}
    missing = sorted(expected.keys() - actual.keys())
    extra = sorted(actual.keys() - expected.keys())
    mismatched = sorted(
        k for k in expected.keys() & actual.keys() if expected[k] != actual[k]
    )
    report = []
    for k in missing:
        report.append(f"missing artifact: {k} (reference {expected[k]})")
    for k in extra:
        report.append(f"unexpected artifact: {k} (actual {actual[k]})")
    for k in mismatched:
        report.append(
            f"digest mismatch: {k} (reference {expected[k]}, actual {actual[k]})"
        )
    assert not report, f"bit-identity violation in '{prefix}': " + "; ".join(report)


@pytest.mark.parametrize("suite", MOCK_SUITES)
def test_mock_suite_bitidentity(battery: Battery, suite: str) -> None:
    if _RECORD:
        pytest.skip("record mode: manifest rewritten, nothing to verify")
    _compare_keys(battery.artifacts, suite)


@pytest.mark.parametrize("sim", list(SIMS))
def test_sim_bitidentity(battery: Battery, sim: str) -> None:
    if _RECORD:
        pytest.skip("record mode: manifest rewritten, nothing to verify")
    _compare_keys(battery.artifacts, sim)


@pytest.mark.parametrize("sim", list(SIMS))
def test_pinned_final_metrics(battery: Battery, sim: str) -> None:
    """Final conservation block matches the literal gate acceptance numbers."""
    stdout = battery.conservation.get(sim)
    assert stdout is not None
    # last conservation block: the final values of each printed quantity
    for label, expected in PINNED_METRICS[sim]:
        matches = re.findall(rf"{re.escape(label)} = (\S+)", stdout)
        assert matches, f"{sim}: no '{label}' line found"
        assert matches[-1] == expected, (
            f"{sim}: final '{label}' = {matches[-1]}, expected {expected}"
        )


def test_no_nan_in_conservation_logs(battery: Battery) -> None:
    """No NaN anywhere in the conservation/physics log lines of both sims."""
    for sim, stdout in battery.conservation.items():
        physics_lines = [
            line
            for line in stdout.splitlines()
            if "AMR " in line or "conservation" in line
        ]
        assert physics_lines, f"{sim}: no AMR conservation lines found"
        for line in physics_lines:
            assert "nan" not in line.lower(), f"{sim}: NaN in log line: {line}"
