#!/usr/bin/env python3
# ruff: noqa: ANN401
"""Validate output data produced by test_outputdata.

Supports both BP5 file and SST streaming engines. The script checks that
3D, 3Dcut, and 2D outputs contain the expected variables, have plausible
shapes, and hold finite values inside physically reasonable ranges.
"""

import argparse
import os
import pathlib
import shutil
import subprocess
import sys
import tempfile
import time
from typing import Any

import numpy as np

import adios2

EXPECTED_VARIABLES = [
    "lbm_density",
    "lbm_density_fluctuation",
    "velocity_x",
    "velocity_y",
    "velocity_z",
]

# Tight physical bounds for the quantities written by test_outputdata.
# density_fluctuation is defined as density - 1, so the two ranges are kept
# consistent. The inflow is along x, so vy and vz are constrained more tightly
# than vx.
VALUE_BOUNDS = {
    "lbm_density": (0.9, 1.2),
    "lbm_density_fluctuation": (-0.1, 0.2),
    "velocity_x": (-0.2, 0.2),
    "velocity_y": (-0.1, 0.1),
    "velocity_z": (-0.1, 0.1),
}

OUTPUTS = {
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


def find_results_directory(project_dir: pathlib.Path) -> pathlib.Path:
    """Find the results directory created by test_outputdata."""
    candidates = sorted(project_dir.glob("results_test_outputdata_*"))
    if not candidates:
        raise RuntimeError(
            "No results directory matching 'results_test_outputdata_*' found"
        )
    if len(candidates) > 1:
        print(
            "Warning: multiple results directories found, "
            f"using the first: {candidates[0]}"
        )
    return candidates[0]


def wait_for_results_directory(
    project_dir: pathlib.Path, timeout_sec: float = 60.0
) -> pathlib.Path:
    """Wait for the results directory to appear (needed for SST)."""
    deadline = time.monotonic() + timeout_sec
    while time.monotonic() < deadline:
        candidates = sorted(project_dir.glob("results_test_outputdata_*"))
        if candidates:
            return candidates[0]
        time.sleep(0.1)
    raise RuntimeError("Timed out waiting for results directory")


def cleanup_results_directory(project_dir: pathlib.Path) -> None:
    """Remove any leftover results directory from a previous run."""
    for candidate in project_dir.glob("results_test_outputdata_*"):
        if candidate.is_dir():
            print(f"Removing leftover results directory: {candidate}")
            shutil.rmtree(candidate)


def check_variable(
    name: str,
    var: Any | None,
    expected_kind: str,
    bp_path: pathlib.Path,
) -> list[int]:
    """Validate variable metadata and return its shape."""
    if var is None:
        raise RuntimeError(f"Variable {name} not found in {bp_path}")

    shape = list(var.shape())

    if len(shape) != 3:
        raise RuntimeError(
            f"Variable {name} in {bp_path} has {len(shape)} dimensions, expected 3"
        )

    if any(dim <= 0 for dim in shape):
        raise RuntimeError(
            f"Variable {name} in {bp_path} has non-positive shape: {shape}"
        )

    unit_dims = sum(1 for dim in shape if dim == 1)
    if expected_kind == "3D" and unit_dims != 0:
        raise RuntimeError(
            f"Variable {name} in {bp_path} is not a full 3D array: {shape}"
        )
    if expected_kind == "2D" and unit_dims != 1:
        raise RuntimeError(
            f"Variable {name} in {bp_path} is not a 2D cut (one unit axis): {shape}"
        )

    return shape


def check_value_range(
    name: str, data_min: float, data_max: float, bp_path: pathlib.Path
) -> None:
    """Check that min/max values are finite and inside the expected bounds."""
    if not np.isfinite(data_min) or not np.isfinite(data_max):
        raise RuntimeError(f"Variable {name} in {bp_path} contains NaN or Inf")

    lo, hi = VALUE_BOUNDS[name]
    print(f"    value range: [{data_min:.6g}, {data_max:.6g}] (expected [{lo}, {hi}])")
    if data_min < lo or data_max > hi:
        raise RuntimeError(
            f"Variable {name} in {bp_path} is out of bounds: "
            f"[{data_min:.6g}, {data_max:.6g}] not in [{lo}, {hi}]"
        )


def check_bp5_file(bp_path: pathlib.Path, expected_kind: str) -> None:
    """Validate a single BP5 file."""
    print(f"Checking {bp_path} ...")
    if not bp_path.exists():
        raise RuntimeError(f"Output file does not exist: {bp_path}")

    with adios2.FileReader(str(bp_path)) as reader:
        variables = reader.available_variables()
        num_steps = reader.num_steps()
        print(f"  steps: {num_steps}")

        if num_steps == 0:
            raise RuntimeError(f"No steps written to {bp_path}")

        missing = [name for name in EXPECTED_VARIABLES if name not in variables]
        if missing:
            raise RuntimeError(f"Missing variables in {bp_path}: {missing}")

        for name in EXPECTED_VARIABLES:
            var = reader.inquire_variable(name)
            shape = check_variable(name, var, expected_kind, bp_path)
            print(f"  {name}: shape={shape}")

            data = reader.read(var, step_selection=[0, 1])
            assert data is not None
            if data.size == 0:
                raise RuntimeError(f"Variable {name} in {bp_path} has no data")

            check_value_range(name, float(data.min()), float(data.max()), bp_path)


def _read_sst_step(
    engine: Any,
    io: Any,
    stream_path: pathlib.Path,
    expected_kind: str,
    step_count: int,
    value_mins: dict[str, float],
    value_maxs: dict[str, float],
) -> None:
    """Read all expected variables from the current step of an SST engine."""
    for name in EXPECTED_VARIABLES:
        var = io.inquire_variable(name)
        if step_count == 1:
            shape = check_variable(name, var, expected_kind, stream_path)
            print(f"  {name}: shape={shape}")

        if var is None:
            raise RuntimeError(
                f"Variable {name} not found in {stream_path} step {step_count}"
            )

        shape = list(var.shape())
        dtype = np.float32 if var.type() == "float" else np.float64
        data = np.zeros(shape, dtype=dtype)
        engine.get(var, data)
        if data.size == 0:
            raise RuntimeError(f"Variable {name} in {stream_path} has no data")

        data_min = float(data.min())
        data_max = float(data.max())
        value_mins[name] = min(value_mins.get(name, data_min), data_min)
        value_maxs[name] = max(value_maxs.get(name, data_max), data_max)


def check_sst_stream(
    stream_path: pathlib.Path, expected_kind: str, adios_config: pathlib.Path
) -> None:
    """Validate a single SST stream."""
    print(f"Checking {stream_path} ...")

    step_count = 0
    value_mins: dict[str, float] = {}
    value_maxs: dict[str, float] = {}

    Adios = getattr(adios2, "Adios")
    Mode = getattr(adios2, "Mode")
    StepStatus = getattr(adios2, "StepStatus")
    adios = Adios(str(adios_config))
    io = adios.declare_io("Output")

    with io.open(str(stream_path), Mode.Read) as engine:
        status = engine.begin_step()
        if status != StepStatus.OK:
            raise RuntimeError(f"No steps received from {stream_path}")

        variables = io.available_variables()
        missing = [name for name in EXPECTED_VARIABLES if name not in variables]
        if missing:
            raise RuntimeError(f"Missing variables in {stream_path}: {missing}")

        while True:
            step_count += 1
            _read_sst_step(
                engine,
                io,
                stream_path,
                expected_kind,
                step_count,
                value_mins,
                value_maxs,
            )

            engine.end_step()
            status = engine.begin_step()
            if status != StepStatus.OK:
                break

    print(f"  steps: {step_count}")
    if step_count == 0:
        raise RuntimeError(f"No steps received from {stream_path}")

    for name in EXPECTED_VARIABLES:
        check_value_range(name, value_mins[name], value_maxs[name], stream_path)


def _begin_all_sst_steps(
    engines: list,
    ios: list,
    stream_paths: list[tuple[pathlib.Path, str]],
    StepStatus: type,
) -> None:
    """Begin the first step on every engine and verify expected variables."""
    for idx, engine in enumerate(engines):
        status = engine.begin_step()
        if status != StepStatus.OK:
            raise RuntimeError(f"No steps received from {stream_paths[idx][0]}")

        variables = ios[idx].available_variables()
        missing = [name for name in EXPECTED_VARIABLES if name not in variables]
        if missing:
            raise RuntimeError(
                f"Missing variables in {stream_paths[idx][0]}: {missing}"
            )


def _read_sst_streams_round_robin(
    engines: list,
    ios: list,
    stream_paths: list[tuple[pathlib.Path, str]],
    step_counts: list[int],
    value_mins: list[dict[str, float]],
    value_maxs: list[dict[str, float]],
    StepStatus: type,
) -> None:
    """Consume one step from each engine per iteration until all streams end."""
    while True:
        any_active = False
        for idx, engine in enumerate(engines):
            stream_path, expected_kind = stream_paths[idx]
            step_counts[idx] += 1
            _read_sst_step(
                engine,
                ios[idx],
                stream_path,
                expected_kind,
                step_counts[idx],
                value_mins[idx],
                value_maxs[idx],
            )
            engine.end_step()
            status = engine.begin_step()
            if status == StepStatus.OK:
                any_active = True

        if not any_active:
            break


def _check_sst_value_ranges(
    stream_paths: list[tuple[pathlib.Path, str]],
    step_counts: list[int],
    value_mins: list[dict[str, float]],
    value_maxs: list[dict[str, float]],
) -> None:
    """Print step counts and validate aggregated value ranges."""
    for idx, (stream_path, _) in enumerate(stream_paths):
        print(f"  {stream_path}: steps={step_counts[idx]}")
        if step_counts[idx] == 0:
            raise RuntimeError(f"No steps received from {stream_path}")

    for idx, (stream_path, _) in enumerate(stream_paths):
        for name in EXPECTED_VARIABLES:
            check_value_range(
                name, value_mins[idx][name], value_maxs[idx][name], stream_path
            )


def check_sst_streams(
    stream_paths: list[tuple[pathlib.Path, str]], adios_config: pathlib.Path
) -> None:
    """Validate multiple SST streams in parallel to avoid writer/reader deadlocks."""
    if not stream_paths:
        return

    Adios = getattr(adios2, "Adios")
    Mode = getattr(adios2, "Mode")
    StepStatus = getattr(adios2, "StepStatus")
    adios = Adios(str(adios_config))

    # Each stream needs its own IO object because all streams define a "TIME"
    # variable and sharing one IO would cause variable definition conflicts.
    # The per-stream IOs are not named "Output" in the config, so they inherit
    # the SST engine and parameters from the configured "Output" IO.
    default_io = adios.declare_io("Output")
    engines: list = []
    ios: list = []
    for stream_path, _ in stream_paths:
        print(f"Checking {stream_path} ...")
        io = adios.declare_io(f"ReaderIO_{stream_path.name}")
        io.set_engine(default_io.engine_type())
        io.set_parameters(default_io.parameters())
        engine = io.open(str(stream_path), Mode.Read)
        engines.append(engine)
        ios.append(io)

    try:
        step_counts = [0] * len(stream_paths)
        value_mins: list[dict[str, float]] = [{} for _ in stream_paths]
        value_maxs: list[dict[str, float]] = [{} for _ in stream_paths]

        _begin_all_sst_steps(engines, ios, stream_paths, StepStatus)
        _read_sst_streams_round_robin(
            engines, ios, stream_paths, step_counts, value_mins, value_maxs, StepStatus
        )
        _check_sst_value_ranges(stream_paths, step_counts, value_mins, value_maxs)
    finally:
        for engine in engines:
            engine.close()


def run_simulation(
    simulation: pathlib.Path,
    adios_config: pathlib.Path,
    output_kind: str,
    project_dir: pathlib.Path,
    resolution: int,
) -> subprocess.Popen:
    """Launch the test simulation as a background subprocess."""
    cmd = [
        str(simulation),
        "--adios-config",
        str(adios_config),
        "--output-kind",
        output_kind,
        "--resolution",
        str(resolution),
    ]
    env = os.environ.copy()
    # On nodes where both ROCm and CUDA are visible to Open MPI, select CUDA.
    env.setdefault("OMPI_MCA_accelerator", "cuda")
    print(f"Starting simulation: {' '.join(cmd)}")
    return subprocess.Popen(cmd, cwd=project_dir, env=env)


def validate_bp5(
    project_dir: pathlib.Path,
    simulation: pathlib.Path,
    adios_config: pathlib.Path,
    output_kind: str,
    resolution: int,
) -> None:
    """Run the simulation and validate the produced BP5 files."""
    proc = run_simulation(
        simulation, adios_config, output_kind, project_dir, resolution
    )
    try:
        returncode = proc.wait()
        if returncode != 0:
            raise RuntimeError(f"Simulation failed with exit code {returncode}")
    finally:
        if proc.poll() is None:
            proc.terminate()
            proc.wait()

    results_dir = find_results_directory(project_dir)
    print(f"Using results directory: {results_dir}")

    for output_name, kind in OUTPUTS[output_kind]:
        check_bp5_file(results_dir / f"{output_name}.bp", kind)


def validate_sst(
    project_dir: pathlib.Path,
    simulation: pathlib.Path,
    adios_config: pathlib.Path,
    output_kind: str,
    resolution: int,
) -> None:
    """Launch the simulation and validate the produced SST streams."""
    proc = run_simulation(
        simulation, adios_config, output_kind, project_dir, resolution
    )
    try:
        results_dir = wait_for_results_directory(project_dir)
        print(f"Using results directory: {results_dir}")

        stream_paths = [
            (results_dir / output_name, kind)
            for output_name, kind in OUTPUTS[output_kind]
        ]
        # Single-stream output categories can be read sequentially; multi-stream
        # 2D cuts must be opened and consumed together to prevent deadlocks
        # with SST's QueueLimit=1 / QueueFullPolicy=Block configuration.
        if len(stream_paths) == 1:
            check_sst_stream(stream_paths[0][0], stream_paths[0][1], adios_config)
        else:
            check_sst_streams(stream_paths, adios_config)

        returncode = proc.wait()
        if returncode != 0:
            raise RuntimeError(f"Simulation failed with exit code {returncode}")
    finally:
        if proc.poll() is None:
            proc.terminate()
            proc.wait()


def validate_inline(
    project_dir: pathlib.Path,
    simulation: pathlib.Path,
    adios_config: pathlib.Path,
    output_kind: str,
    resolution: int,
) -> None:
    """Launch the simulation with the Inline/Plugin engine.

    Verifies that the Catalyst pipeline executes at least one step and that the
    Fides data model file is generated.
    """
    # The Inline/Plugin engine needs Catalyst/ADIOS2 plugin environment set.
    env = os.environ.copy()
    # On nodes where both ROCm and CUDA are visible to Open MPI, select CUDA.
    env.setdefault("OMPI_MCA_accelerator", "cuda")
    env.setdefault("ADIOS2_PLUGIN_PATH", "/usr/lib")
    env.setdefault("CATALYST_IMPLEMENTATION_PATHS", "/usr/lib/catalyst")
    env.setdefault("CATALYST_IMPLEMENTATION_NAME", "paraview")
    env.setdefault("TNL_LBM_HEADLESS", "1")

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".txt", dir=project_dir, delete=False
    ) as marker_file:
        marker_path = pathlib.Path(marker_file.name)

    env["TNL_LBM_CATALYST_STEPS_FILE"] = str(marker_path)

    cmd = [
        str(simulation),
        "--adios-config",
        str(adios_config),
        "--output-kind",
        output_kind,
        "--resolution",
        str(resolution),
    ]
    print(f"Starting simulation: {' '.join(cmd)}")
    proc = subprocess.Popen(cmd, cwd=project_dir, env=env)
    try:
        returncode = proc.wait()
        if returncode != 0:
            raise RuntimeError(f"Simulation failed with exit code {returncode}")
    finally:
        if proc.poll() is None:
            proc.terminate()
            proc.wait()

    results_dir = find_results_directory(project_dir)
    print(f"Using results directory: {results_dir}")

    fides_model = results_dir / "lbm-fides.json"
    if not fides_model.exists():
        raise RuntimeError(f"Fides data model not generated: {fides_model}")

    if not marker_path.exists():
        raise RuntimeError(f"Catalyst step marker not written: {marker_path}")

    content = marker_path.read_text().strip()
    try:
        steps = int(content)
    except ValueError as exc:
        raise RuntimeError(
            f"Catalyst step marker has invalid content: {content!r}"
        ) from exc

    print(f"  Catalyst executed {steps} step(s)")
    if steps == 0:
        raise RuntimeError("Catalyst pipeline did not execute any steps")

    marker_path.unlink(missing_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate output data from test_outputdata"
    )
    parser.add_argument(
        "--project-dir",
        type=pathlib.Path,
        default=pathlib.Path.cwd(),
        help="project directory where results_* are written (default: cwd)",
    )
    parser.add_argument(
        "--engine",
        choices=["bp5", "sst", "inline"],
        default="bp5",
        help="ADIOS2 engine to validate (default: bp5)",
    )
    parser.add_argument(
        "--simulation",
        type=pathlib.Path,
        default=pathlib.Path("build/tests/test_outputdata"),
        help="path to the test_outputdata executable",
    )
    parser.add_argument(
        "--adios-config",
        type=pathlib.Path,
        default=None,
        help=(
            "path to the ADIOS2 config file"
            " (default: adios2.xml for bp5, adios2_sst.xml for sst,"
            " tests/adios2-inline-plugin.xml for inline)"
        ),
    )
    parser.add_argument(
        "--output-kind",
        choices=["all", "3d", "3dcut", "2d"],
        default="all",
        help="which outputs to validate (default: all)",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=1,
        help="lattice resolution passed to the simulation (default: 1)",
    )
    args = parser.parse_args()

    project_dir = args.project_dir.resolve()
    simulation = args.simulation.resolve()
    if args.adios_config:
        adios_config = args.adios_config.resolve()
    elif args.engine == "bp5":
        adios_config = project_dir / "adios2.xml"
    elif args.engine == "sst":
        adios_config = project_dir / "adios2_sst.xml"
    else:
        adios_config = project_dir / "tests" / "adios2-inline-plugin.xml"
    output_kind = args.output_kind
    resolution = args.resolution

    cleanup_results_directory(project_dir)

    if args.engine == "bp5":
        validate_bp5(project_dir, simulation, adios_config, output_kind, resolution)
    elif args.engine == "sst":
        validate_sst(project_dir, simulation, adios_config, output_kind, resolution)
    else:
        validate_inline(project_dir, simulation, adios_config, output_kind, resolution)

    print("All output-data checks passed.")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except RuntimeError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
