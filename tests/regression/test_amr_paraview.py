"""ParaView end-to-end checks of the AMR VTKHDF output (pytest launcher).

Both arms load a VTKHDF OverlappingAMR file in ParaView (>= 6.0) through
the VTKHDF reader, verify the AMR level structure, cell fields and values,
and render a non-trivial PNG — the real-consumer complement of the
``amr_vtkhdf_writer`` doctest suite (pytest-only counterparts of the
retired ``tests/test_amr_paraview_e2e_{,_nesting}.sh`` wrappers):

- ``test_amr_paraview_e2e``: ``sim_AMR --resolution 1`` data (2 levels),
  driven by ``tests/amr_paraview_e2e.py`` under pvpython;
- ``test_amr_paraview_e2e_nesting``: the 3-level telescoping chain of the
  dedicated mock ``build/tests/test_amr_nesting_sim`` (4 levels), driven
  by ``tests/amr_paraview_e2e_nesting.py`` under pvpython.

The input data is regenerated into the session workspace on every pytest
run (the retired shells reused results_*/ in the project root instead —
hermetic per-run state is the pytest convention). Both arms skip when
pvpython is not on PATH (the shells' exit-77 convention), and fail with a
build hint when the producing binary is missing.
"""

from __future__ import annotations

import pathlib
import shutil

import pytest

from tests.lbmtest import ADIOS_CONFIG, BUILD_DIR, PROJECT_ROOT, run_sim

_PVPYTHON = shutil.which("pvpython")

pytestmark = pytest.mark.skipif(
    _PVPYTHON is None,
    reason="pvpython not found on PATH (needs ParaView >= 6.0)",
)


def _regen_arm(
    workspace: pathlib.Path,
    name: str,
    binary: pathlib.Path,
    args: list[str],
    target_hint: str,
    results_dir_name: str,
    adios_via_cwd: bool = False,
) -> pathlib.Path:
    """Drive one data-producing sim in a fresh workdir; return the vtkhdf."""
    if not binary.is_file():
        pytest.fail(
            f"cannot find {binary} — build it first: "
            f"cmake --build {BUILD_DIR} --target {target_hint}",
            pytrace=False,
        )
    workdir = workspace / name
    workdir.mkdir()
    # argparse-free drivers (test_amr_nesting_sim) read adios2.xml from cwd
    # via their State ctor; the argparse-driven sims take --adios-config
    if adios_via_cwd:
        shutil.copy(ADIOS_CONFIG, workdir / "adios2.xml")
        cmd = [str(binary), *args]
    else:
        cmd = [str(binary), *args, "--adios-config", str(ADIOS_CONFIG)]
    stdout = run_sim(cmd, workdir=workdir, timeout=600.0)
    # SimUpdate announces its natural end on the sim log (stdout) and/or the
    # per-rank log file -- the same dual-channel check the retired shells ran
    main_log = workdir / results_dir_name / "log_main_rank000"
    assert "physFinalTime reached" in stdout or (
        main_log.is_file() and "physFinalTime reached" in main_log.read_text()
    ), f"{name} finished without 'physFinalTime reached'"
    vtkhdf = workdir / results_dir_name / "output_amr_0000.vtkhdf"
    assert vtkhdf.is_file(), f"{name} did not produce {vtkhdf}"
    return vtkhdf


def _pv_run(
    workspace: pathlib.Path, name: str, script: str, vtkhdf: pathlib.Path
) -> str:
    # module-level skipif guarantees pvpython exists for every collected test
    assert _PVPYTHON is not None
    outdir = workspace / f"{name}_render"
    outdir.mkdir()
    return run_sim(
        [
            _PVPYTHON,
            str(PROJECT_ROOT / "tests" / script),
            "--input",
            str(vtkhdf),
            "--outdir",
            str(outdir),
        ],
        workdir=workspace,
        timeout=600.0,
    )


def test_amr_paraview_e2e(workspace: pathlib.Path) -> None:
    vtkhdf = _regen_arm(
        workspace,
        "e2e",
        BUILD_DIR / "sim_AMR" / "sim_AMR",
        ["--resolution", "1"],
        "sim_AMR",
        "results_sim_AMR_res01_np001",
    )
    stdout = _pv_run(workspace, "e2e", "amr_paraview_e2e.py", vtkhdf)
    assert "RESULT: all ParaView E2E checks passed" in stdout


def test_amr_paraview_e2e_nesting(workspace: pathlib.Path) -> None:
    vtkhdf = _regen_arm(
        workspace,
        "e2e_nesting",
        BUILD_DIR / "tests" / "test_amr_nesting_sim",
        [],
        "test_amr_nesting_sim",
        "results_test_amr_nesting_sim_np001",
        adios_via_cwd=True,
    )
    stdout = _pv_run(workspace, "e2e_nesting", "amr_paraview_e2e_nesting.py", vtkhdf)
    assert "RESULT: all ParaView E2E nesting checks passed" in stdout
