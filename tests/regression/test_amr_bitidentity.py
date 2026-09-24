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
  the dataset content of every ``*.vtkhdf`` file they leave behind
  (``amr_vtkhdf_writer`` writes ``test_amr.vtkhdf`` and
  ``test_amr_nesting.vtkhdf``, kept after a green run for exactly this
  pinning);
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


def _single_pattern_aa_tree() -> bool:
    """Single-pattern AA build tree: the AB-pinned gate binaries are
    intentionally not built (the ``AMR_TEST_PATTERNS`` selection of
    tests/unit/CMakeLists.txt), so the ``*_ab`` mock-suite artifacts skip;
    the ``*_aa`` binaries and the sim_AMR runs verify against the same
    manifest there (mL1 production is bitwise-exact across streaming
    patterns — results_drag_crisis/aa_seed_report.md §2)."""
    return (
        not (BUILD_DIR / "tests" / "test_amr_units_ab").is_file()
        and (BUILD_DIR / "tests" / "test_amr_units_aa").is_file()
    )


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
        ("AMR level 1: kinetic energy", "1.832124e+00"),
    ],
    "sim_AMR_channel": [
        ("AMR conservation: mass", "1.866006e+04"),
        ("AMR level 0: kinetic energy", "8.411522e+01"),
        ("AMR level 1: kinetic energy", "5.097541e+01"),
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
        if " RAM for DFs:" in line:
            # DF-buffer footprint follows the pattern's DFMAX (single- vs
            # two-array trees), not the physics
            continue
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
            if not _RECORD and suite.endswith("_ab") and _single_pattern_aa_tree():
                # single-pattern AA tree: the AB-pinned suites contribute no
                # artifacts; the parametrized comparison skips on the same arm
                continue
            pytest.fail(
                f"cannot find {binary} — build the AMR test targets first: "
                f"cmake --build {BUILD_DIR} --target {binary_name}",
                pytrace=False,
            )
        workdir = root / suite
        workdir.mkdir()
        # the suites read adios2.xml from their cwd (State ctor); --success
        # makes doctest print every passing assertion's message: the
        # FP-metric texts the pre-port PASS lines carried stay inside the
        # pinned stream (they are how this harness detects sub-tolerance
        # FP drift -- the doctest banner alone is a mere count)
        shutil.copy(ADIOS_CONFIG, workdir / "adios2.xml")
        stdout = run_sim(
            [
                binary,
                f"--test-suite={suite_filter}",
                "--no-colors",
                "--no-duration",
                "--success",
            ],
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
                "short runs; re-recorded after the rebase onto main (TNL 0.4.2, "
                "symmetry-corner and storage-aware BC changes shift the ulp "
                "baseline); re-recorded again on 2026-09-18 for the "
                "setInitialCondition-engine IC port (two era-parity deltas: the "
                "sim_AMR frame-0000 t=0 macro provenance — engine IC values vs "
                "legacy recompute-from-DFs, ulp-level at 262076/262144 TG sites, "
                "all production frames/stdout proven byte-identical by the "
                "temp-computeInitialMacro reproduction experiment — and the "
                "subcycling_ab Test-4 SB1 baseline statistic, whose fine "
                "frame-1 inner-ghost init residue is now IC-stamped instead of "
                "zero-init) and verified by "
                "tests/regression/test_amr_bitidentity.py; re-recorded again on "
                "2026-09-18: SimInit interpolation-recompute replaced by "
                "IC-sourced t=0 ring macros (analytic per-level IC authored over "
                "the footprint ghost window); era-parity delta confined to "
                "sim_AMR frame-0000 (engine-IC interior provenance + analytic "
                "ring; channel bytes unchanged); re-recorded again on "
                "2026-09-18 after the setInitialCondition owned-site macro "
                "guard was dropped engine-wide (ghost-row macros now authored "
                "from the initial condition everywhere, retiring the AMR-local "
                "ring stamping and the writer suite's recompute helper) — only "
                "the two writer-suite stdout digests moved (doctest line "
                "references shifted by the cleanup); re-recorded again on "
                "2026-09-24 after the esoteric-pattern port of the gate suites "
                "shifted doctest line references in test_amr_coupling.cu — "
                "only the two coupling-suite stdout digests moved"
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


# sim_AMR_channel at --resolution 1 runs the R=1 inflow-adjacent geometry
# whose C2F wall-guard path is a DOCUMENTED known defect under the
# single-array streaming patterns (results_drag_crisis/aa_seed_report.md §6:
# constant-density seed on the L0 inflow plane at cycle 3, chaotic by cycle
# 8), so its artifacts and pinned metrics cannot verify against the AB-tree
# manifest in a single-pattern AA tree until the defect is repaired;
# strict=True makes a future repair fail loudly and force re-enabling.
# sim_AMR's TGV stays verifying: its cross-tree production is proven
# bitwise-exact on the mL1 class (seed report §2)
def _sim_params() -> list:
    params = []
    for sim in SIMS:
        marks = []
        if sim == "sim_AMR_channel" and not _RECORD and _single_pattern_aa_tree():
            marks.append(
                pytest.mark.xfail(
                    reason=(
                        "known defect (results_drag_crisis/aa_seed_report.md "
                        "§6): the R=1 inflow-adjacent C2F wall-guard path "
                        "diverges under the single-array streaming patterns; "
                        "sim_AMR_channel at --resolution 1 runs exactly this "
                        "geometry, so the arm cannot verify against the "
                        "AB-tree manifest in a single-pattern AA tree until "
                        "repaired"
                    ),
                    strict=True,
                )
            )
        params.append(pytest.param(sim, marks=marks))
    return params


@pytest.mark.parametrize("suite", MOCK_SUITES)
def test_mock_suite_bitidentity(battery: Battery, suite: str) -> None:
    if _RECORD:
        pytest.skip("record mode: manifest rewritten, nothing to verify")
    if suite.endswith("_ab") and _single_pattern_aa_tree():
        pytest.skip(
            "single-pattern AA tree: the AB-pinned gate binaries are "
            "intentionally not built (AMR_TEST_PATTERNS = "
            "aa eso_twist eso_pull eso_push)"
        )
    _compare_keys(battery.artifacts, suite)


@pytest.mark.parametrize("sim", _sim_params())
def test_sim_bitidentity(battery: Battery, sim: str) -> None:
    if _RECORD:
        pytest.skip("record mode: manifest rewritten, nothing to verify")
    # the "." suffix keeps the sim_AMR prefix out of sim_AMR_channel's keys
    _compare_keys(battery.artifacts, f"{sim}.")


@pytest.mark.parametrize("sim", _sim_params())
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
