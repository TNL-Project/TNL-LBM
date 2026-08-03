"""Integration tests for the output-data pipeline (test_outputdata).

Validates that the simulation's ADIOS2 outputs — BP5 files, SST streams, and
the Inline/Plugin (Catalyst) engine — contain the expected variables with
plausible shapes and finite values inside physically reasonable ranges.

BP5 outputs are validated after the simulation finishes; SST streams are
consumed while the simulation runs (round-robin across streams to avoid
writer/reader deadlocks); the Inline engine is checked via the Catalyst step
marker file and the generated Fides data model.
"""

from __future__ import annotations

import os
import pathlib
import subprocess
import time
from typing import TYPE_CHECKING, Protocol

import adios2
import numpy as np
import pytest

from tests.lbmtest import BUILD_DIR, PROJECT_ROOT

if TYPE_CHECKING:
    from collections.abc import Sequence

SIMULATION = BUILD_DIR / "tests" / "test_outputdata"
ADI_CONFIG_BP5 = PROJECT_ROOT / "adios2.xml"
ADI_CONFIG_SST = PROJECT_ROOT / "adios2_sst.xml"
ADI_CONFIG_INLINE = PROJECT_ROOT / "tests" / "integration" / "adios2-inline-plugin.xml"
PIPELINE_SCRIPT = PROJECT_ROOT / "tests" / "integration" / "catalyst-pipeline.py"

SIM_TIMEOUT = 900.0
SST_DIR_TIMEOUT = 60.0

EXPECTED_VARIABLES = [
    "lbm_density",
    "lbm_density_fluctuation",
    "velocity_x",
    "velocity_y",
    "velocity_z",
]

# Tight physical bounds for the quantities written by test_outputdata.
# density_fluctuation is defined as density - 1, so the two ranges are kept
# consistent. The inflow is along x, so vy and vz are constrained more
# tightly than vx.
VALUE_BOUNDS = {
    "lbm_density": (0.9, 1.2),
    "lbm_density_fluctuation": (-0.1, 0.2),
    "velocity_x": (-0.2, 0.2),
    "velocity_y": (-0.1, 0.1),
    "velocity_z": (-0.1, 0.1),
}

OUTPUTS: dict[str, list[tuple[str, str]]] = {
    "all": [
        ("output_3D", "3D"),
        ("output_3Dcut_box", "3D"),
        ("output_2D_cut_X", "2D"),
        ("output_2D_cut_Y", "2D"),
        ("output_2D_cut_Z", "2D"),
    ],
    "3d": [("output_3D", "3D")],
    "3dcut": [("output_3Dcut_box", "3D")],
    "2d": [
        ("output_2D_cut_X", "2D"),
        ("output_2D_cut_Y", "2D"),
        ("output_2D_cut_Z", "2D"),
    ],
}

StreamSpec = tuple[pathlib.Path, str]


class AdiosVariable(Protocol):
    def shape(self) -> Sequence[int]: ...
    def type(self) -> str: ...


class AdiosEngine(Protocol):
    def begin_step(self) -> object: ...
    def end_step(self) -> None: ...
    def get(
        self, variable: AdiosVariable, content: np.ndarray | None = ...
    ) -> None: ...
    def close(self) -> None: ...


class AdiosIO(Protocol):
    def inquire_variable(self, name: str) -> AdiosVariable | None: ...
    def set_engine(self, engine_type: object) -> None: ...
    def set_parameters(self, parameters: dict[str, str]) -> None: ...
    def open(self, name: str, mode: object) -> AdiosEngine: ...
    def available_variables(self) -> dict[str, object]: ...
    def engine_type(self) -> object: ...
    def parameters(self) -> dict[str, str]: ...


def sim_env() -> dict[str, str]:
    """Environment for test_outputdata runs."""
    env = os.environ.copy()
    # On nodes where both ROCm and CUDA are visible to Open MPI, select CUDA.
    env.setdefault("OMPI_MCA_accelerator", "cuda")
    return env


def launch_simulation(
    adios_config: pathlib.Path, output_kind: str, resolution: int, workdir: pathlib.Path
) -> subprocess.Popen[bytes]:
    """Launch test_outputdata as a background process in workdir."""
    cmd = [
        str(SIMULATION),
        "--adios-config",
        str(adios_config),
        "--output-kind",
        output_kind,
        "--resolution",
        str(resolution),
    ]
    return subprocess.Popen(cmd, cwd=workdir, env=sim_env())


def wait_simulation(proc: subprocess.Popen[bytes]) -> None:
    """Wait for the simulation to finish; kill it on timeout and fail."""
    try:
        returncode = proc.wait(timeout=SIM_TIMEOUT)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()
        pytest.fail(
            f"test_outputdata timed out after {SIM_TIMEOUT:.0f}s", pytrace=False
        )
    if returncode != 0:
        pytest.fail(
            f"test_outputdata failed with exit code {returncode}", pytrace=False
        )


def find_results_directory(workdir: pathlib.Path) -> pathlib.Path:
    """Find the (single) results directory created by test_outputdata."""
    candidates = sorted(workdir.glob("results_test_outputdata_*"))
    assert candidates, "no results directory matching 'results_test_outputdata_*' found"
    if len(candidates) > 1:
        print(f"Warning: multiple results dirs, using the first: {candidates[0]}")
    return candidates[0]


def wait_for_results_directory(
    workdir: pathlib.Path, timeout_sec: float = SST_DIR_TIMEOUT
) -> pathlib.Path:
    """Wait for the results directory to appear (needed for SST)."""
    deadline = time.monotonic() + timeout_sec
    while time.monotonic() < deadline:
        candidates = sorted(workdir.glob("results_test_outputdata_*"))
        if candidates:
            return candidates[0]
        time.sleep(0.1)
    pytest.fail("timed out waiting for the results directory", pytrace=False)
    raise AssertionError("unreachable")


def check_variable_shape(
    name: str, var: AdiosVariable | None, expected_kind: str, bp_path: pathlib.Path
) -> None:
    """Assert variable metadata (3D block vs 2D cut) matches the output kind."""
    assert var is not None, f"variable {name} not found in {bp_path}"
    shape = list(var.shape())
    assert len(shape) == 3, (
        f"{name} in {bp_path} has {len(shape)} dimensions, expected 3"
    )
    assert all(dim > 0 for dim in shape), (
        f"{name} in {bp_path} has non-positive shape: {shape}"
    )

    unit_dims = sum(1 for dim in shape if dim == 1)
    if expected_kind == "3D":
        assert unit_dims == 0, f"{name} in {bp_path} is not a full 3D array: {shape}"
    else:
        assert unit_dims == 1, (
            f"{name} in {bp_path} is not a 2D cut (one unit axis): {shape}"
        )


def check_value_range(
    name: str, data_min: float, data_max: float, bp_path: pathlib.Path
) -> None:
    """Assert min/max values are finite and inside the expected bounds."""
    assert np.isfinite(data_min) and np.isfinite(data_max), (
        f"{name} in {bp_path} contains NaN or Inf"
    )
    lo, hi = VALUE_BOUNDS[name]
    assert lo <= data_min and data_max <= hi, (
        f"{name} in {bp_path}: [{data_min:.6g}, {data_max:.6g}] not in [{lo}, {hi}]"
    )


def check_bp5_file(bp_path: pathlib.Path, expected_kind: str) -> None:
    """Validate a single BP5 output file."""
    assert bp_path.exists(), f"output file does not exist: {bp_path}"
    with adios2.FileReader(str(bp_path)) as reader:
        assert reader.num_steps() > 0, f"no steps written to {bp_path}"
        variables = reader.available_variables()
        missing = [name for name in EXPECTED_VARIABLES if name not in variables]
        assert not missing, f"missing variables in {bp_path}: {missing}"

        for name in EXPECTED_VARIABLES:
            var = reader.inquire_variable(name)
            check_variable_shape(name, var, expected_kind, bp_path)
            data = reader.read(var, step_selection=[0, 1])
            assert data is not None and data.size > 0, (
                f"{name} in {bp_path} has no data"
            )
            check_value_range(name, float(data.min()), float(data.max()), bp_path)


def read_sst_step(
    engine: AdiosEngine,
    io: AdiosIO,
    stream_path: pathlib.Path,
    expected_kind: str,
    step_count: int,
    value_mins: dict[str, float],
    value_maxs: dict[str, float],
) -> None:
    """Read all expected variables from the current step of an SST engine."""
    for name in EXPECTED_VARIABLES:
        var = io.inquire_variable(name)
        assert var is not None, (
            f"variable {name} not found in {stream_path} step {step_count}"
        )
        if step_count == 1:
            check_variable_shape(name, var, expected_kind, stream_path)

        shape = list(var.shape())
        dtype = np.float32 if var.type() == "float" else np.float64
        data = np.zeros(shape, dtype=dtype)
        engine.get(var, data)
        assert data.size > 0, f"{name} in {stream_path} has no data"

        value_mins[name] = min(value_mins.get(name, np.inf), float(data.min()))
        value_maxs[name] = max(value_maxs.get(name, -np.inf), float(data.max()))


def open_sst_engines(
    streams: Sequence[StreamSpec], adios_config: pathlib.Path
) -> tuple[object, list[AdiosEngine], list[AdiosIO]]:
    """Open one SST engine per stream and return the owning Adios object too.

    Each stream needs its own IO object because all streams define a "TIME"
    variable and sharing one IO would cause variable definition conflicts.
    The per-stream IOs are not named "Output" in the config, so they inherit
    the SST engine and parameters from the configured "Output" IO.

    The Adios object is returned so the caller keeps it alive while the
    engines are pumped (its destruction would invalidate them).
    """
    # adios2.Adios is shadowed by a same-named submodule; getattr reaches the class.
    adios = getattr(adios2, "Adios")(str(adios_config))
    default_io = adios.declare_io("Output")

    engines: list[AdiosEngine] = []
    ios: list[AdiosIO] = []
    for stream_path, _ in streams:
        io = adios.declare_io(f"ReaderIO_{stream_path.name}")
        io.set_engine(default_io.engine_type())
        io.set_parameters(default_io.parameters())
        engines.append(io.open(str(stream_path), adios2.Mode.Read))
        ios.append(io)
    return adios, engines, ios


def pump_sst_streams(
    engines: list[AdiosEngine],
    ios: list[AdiosIO],
    streams: Sequence[StreamSpec],
    value_mins: list[dict[str, float]],
    value_maxs: list[dict[str, float]],
) -> list[int]:
    """Consume one step from each engine per iteration until all streams end."""
    # Begin the first step on every engine and verify expected variables.
    for engine, io, (stream_path, _) in zip(engines, ios, streams, strict=True):
        status = engine.begin_step()
        assert status == adios2.StepStatus.OK, f"no steps received from {stream_path}"
        variables = io.available_variables()
        missing = [name for name in EXPECTED_VARIABLES if name not in variables]
        assert not missing, f"missing variables in {stream_path}: {missing}"

    step_counts = [0] * len(streams)
    active = [True] * len(streams)
    while any(active):
        for idx, engine in enumerate(engines):
            if not active[idx]:
                continue
            stream_path, kind = streams[idx]
            step_counts[idx] += 1
            read_sst_step(
                engine,
                ios[idx],
                stream_path,
                kind,
                step_counts[idx],
                value_mins[idx],
                value_maxs[idx],
            )
            engine.end_step()
            active[idx] = engine.begin_step() == adios2.StepStatus.OK

    for (stream_path, _), count in zip(streams, step_counts, strict=True):
        assert count > 0, f"no steps received from {stream_path}"
    return step_counts


def consume_sst_streams(
    streams: Sequence[StreamSpec], adios_config: pathlib.Path
) -> None:
    """Consume multiple SST streams in parallel to avoid writer/reader deadlocks."""
    _adios, engines, ios = open_sst_engines(streams, adios_config)
    value_mins: list[dict[str, float]] = [{} for _ in streams]
    value_maxs: list[dict[str, float]] = [{} for _ in streams]
    try:
        pump_sst_streams(engines, ios, streams, value_mins, value_maxs)
    finally:
        for engine in engines:
            engine.close()

    for idx, (stream_path, _) in enumerate(streams):
        for name in EXPECTED_VARIABLES:
            check_value_range(
                name, value_mins[idx][name], value_maxs[idx][name], stream_path
            )


def test_bp5_output(test_dir: pathlib.Path) -> None:
    """BP5 file-based output: all outputs written at once, validated post-run."""
    proc = launch_simulation(
        ADI_CONFIG_BP5, output_kind="all", resolution=1, workdir=test_dir
    )
    try:
        wait_simulation(proc)
    finally:
        if proc.poll() is None:
            proc.terminate()
            proc.wait()

    results_dir = find_results_directory(test_dir)
    for output_name, kind in OUTPUTS["all"]:
        check_bp5_file(results_dir / f"{output_name}.bp", kind)


@pytest.mark.parametrize("output_kind", ["3d", "3dcut", "2d"])
def test_sst_output(test_dir: pathlib.Path, output_kind: str) -> None:
    """SST streaming output: streams consumed while the simulation runs."""
    proc = launch_simulation(
        ADI_CONFIG_SST, output_kind=output_kind, resolution=1, workdir=test_dir
    )
    try:
        results_dir = wait_for_results_directory(test_dir)
        streams = [(results_dir / name, kind) for name, kind in OUTPUTS[output_kind]]
        consume_sst_streams(streams, ADI_CONFIG_SST)
        wait_simulation(proc)
    finally:
        if proc.poll() is None:
            proc.terminate()
            proc.wait()


def test_inline_output(test_dir: pathlib.Path) -> None:
    """Inline/Plugin in-situ output: Catalyst executes steps, Fides model written."""
    # The plugin config references the Catalyst script relative to CWD; write a
    # copy with an absolute path so the test can run from the scratch dir.
    adios_config = test_dir / "adios2-inline-plugin.xml"
    adios_config.write_text(
        ADI_CONFIG_INLINE.read_text().replace(
            "tests/integration/catalyst-pipeline.py",
            str(PROJECT_ROOT / "tests" / "integration" / "catalyst-pipeline.py"),
        )
    )
    marker_path = test_dir / "catalyst_steps.txt"

    env = sim_env()
    env.setdefault("ADIOS2_PLUGIN_PATH", "/usr/lib")
    env.setdefault("CATALYST_IMPLEMENTATION_PATHS", "/usr/lib/catalyst")
    env.setdefault("CATALYST_IMPLEMENTATION_NAME", "paraview")
    env.setdefault("TNL_LBM_HEADLESS", "1")
    env["TNL_LBM_CATALYST_STEPS_FILE"] = str(marker_path)

    cmd = [
        str(SIMULATION),
        "--adios-config",
        str(adios_config),
        "--output-kind",
        "3d",
        "--resolution",
        "1",
    ]
    proc = subprocess.Popen(cmd, cwd=test_dir, env=env)
    try:
        wait_simulation(proc)
    finally:
        if proc.poll() is None:
            proc.terminate()
            proc.wait()

    results_dir = find_results_directory(test_dir)
    fides_model = results_dir / "lbm-fides.json"
    assert fides_model.exists(), f"Fides data model not generated: {fides_model}"

    assert marker_path.exists(), f"Catalyst step marker not written: {marker_path}"
    steps = int(marker_path.read_text().strip())
    assert steps > 0, "Catalyst pipeline did not execute any steps"
