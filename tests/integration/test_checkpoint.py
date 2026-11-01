"""End-to-end checkpoint/restart regression tests (test_checkpoint).

In production runs the checkpoint is written when a wall-time limit is
reached, which is not reproducible enough for a regression test. The test
simulation therefore offers an iteration-based interruption: a run stops at
``--checkpoint-after`` iterations and saves the checkpoint, the next run
resumes from it and either goes on to the next checkpoint or finishes.

The restarted run must produce the same final fields as a run that was never
interrupted, bit for bit. That requires the checkpoint to capture not only
the distribution functions and the boundary map, but also the iteration
counter (it selects the streaming parity and the DF buffer rotation) and the
macroscopic fields. Writing the second checkpoint also exercises the staged
write, which replaces the checkpoint of the previous run.
"""

from __future__ import annotations

import pathlib

import numpy as np
import pytest

from tests.lbmtest import ADIOS_CONFIG, BUILD_DIR, FieldData, run_sim
from tests.regression.bp5 import read_last_step

SIMULATION = BUILD_DIR / "tests" / "integration" / "test_checkpoint"

RESOLUTION = 1
ITERATIONS = 200
# The first checkpoint is deliberately at an odd iteration so that the
# single-array streaming patterns (A-A and the esoteric variants) are restored
# from their parity-1/twisted layout, not only from the parity-0 one.
CHECKPOINTS = (41, 80)
FIELDS = ["lbm_density", "velocity_x", "velocity_y", "velocity_z", "wall"]


def results_dir(workdir: pathlib.Path, np_ranks: int) -> pathlib.Path:
    """Path of the results directory written by test_checkpoint."""
    return workdir / f"results_test_checkpoint_res{RESOLUTION:02d}_np{np_ranks:03d}"


def run(workdir: pathlib.Path, np_ranks: int, checkpoint_after: int = 0) -> str:
    """Run test_checkpoint in workdir, optionally interrupting it at a checkpoint."""
    cmd = [
        SIMULATION,
        "--adios-config",
        ADIOS_CONFIG,
        "--resolution",
        str(RESOLUTION),
        "--iterations",
        str(ITERATIONS),
    ]
    if checkpoint_after > 0:
        cmd.extend(["--checkpoint-after", str(checkpoint_after)])
    return run_sim(cmd, workdir=workdir, np_ranks=np_ranks, timeout=300.0)


def read_final_fields(directory: pathlib.Path) -> FieldData:
    """Read the final 3D output written at the end of a completed run."""
    return read_last_step(directory / "output_3D.bp", FIELDS)


def assert_same_fields(actual: FieldData, expected: FieldData) -> None:
    """Assert two field sets are bitwise identical."""
    for name in FIELDS:
        a = actual[name].astype(np.float64)
        b = expected[name].astype(np.float64)
        max_diff = float(np.max(np.abs(a - b))) if a.size else 0.0
        assert np.array_equal(a, b), f"{name} differs after restart: max|diff|={max_diff:.3e}"


@pytest.mark.parametrize("np_ranks", [1, 2], ids=["np1", "np2"])
def test_checkpoint_restart_matches_continuous_run(test_dir: pathlib.Path, np_ranks: int) -> None:
    """Interrupted runs resumed from checkpoints match an uninterrupted run."""
    interrupted_dir = test_dir / "interrupted"
    interrupted_dir.mkdir()
    reference_dir = test_dir / "reference"
    reference_dir.mkdir()
    checkpoint_dir = results_dir(interrupted_dir, np_ranks)

    # First run: stop at the first checkpoint, no final output is written
    stdout = run(interrupted_dir, np_ranks, checkpoint_after=CHECKPOINTS[0])
    assert f"CHECKPOINT_SAVED iteration={CHECKPOINTS[0]}" in stdout
    assert (checkpoint_dir / "checkpoint.bp").is_dir(), "checkpoint was not written"
    assert (checkpoint_dir / "flag.loadstate").exists(), "restart flag was not created"
    assert not (checkpoint_dir / "flag.finished").exists()
    assert not (checkpoint_dir / "output_3D.bp").exists(), "the interrupted run wrote a final output"

    # Second run: resume and overwrite the checkpoint of the previous run
    stdout = run(interrupted_dir, np_ranks, checkpoint_after=CHECKPOINTS[1])
    assert f"Checkpoint loaded successfully: {CHECKPOINTS[0]} iterations" in stdout
    assert f"CHECKPOINT_SAVED iteration={CHECKPOINTS[1]}" in stdout
    assert (checkpoint_dir / "checkpoint.bp").is_dir()

    # Third run: resume from the second checkpoint and finish the simulation
    stdout = run(interrupted_dir, np_ranks)
    assert f"Checkpoint loaded successfully: {CHECKPOINTS[1]} iterations" in stdout
    assert f"SIMULATION_FINISHED iteration={ITERATIONS}" in stdout
    assert (checkpoint_dir / "flag.finished").exists()
    assert not (checkpoint_dir / "flag.loadstate").exists(), "restart flag was not cleared"
    resumed = read_final_fields(checkpoint_dir)

    # Reference run: never interrupted
    run(reference_dir, np_ranks)
    reference = read_final_fields(results_dir(reference_dir, np_ranks))

    assert_same_fields(resumed, reference)

    # A completed results directory must not be recomputed
    stdout = run(interrupted_dir, np_ranks)
    assert "finished state" in stdout
    assert f"SIMULATION_FINISHED iteration={ITERATIONS}" not in stdout
