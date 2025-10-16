#!/usr/bin/env python3
"""
Export ADIOS2 slices to PNG frames.

Example:
    python3 catalyst/tnl_lbm_export_frames.py \
        --instream results_sim_1_res04_np001/output_3D --outdir frames --prefix sim1 --plane yz \
        --varname velocity --component magnitude
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Dict, Mapping, Optional, Sequence, Tuple

import numpy as np

try: 
    from adios2 import Adios, Stream
except ImportError as exc:
    sys.stderr.write(
        "Error: failed to import the 'adios2'\n"
    )
    raise

from tnl_lbm_common import *


def load_matplotlib() -> Tuple[object, object]:
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.gridspec as gridspec  # type: ignore
    import matplotlib.pyplot as plt  # type: ignore

    plt.ioff()
    return plt, gridspec


def io_in_config_file(io_handle) -> bool:
    method = getattr(io_handle, "in_config_file", None)
    if callable(method):
        return bool(method())
    method = getattr(io_handle, "InConfigFile", None)
    if callable(method):
        return bool(method())
    return True


def io_set_engine(io_handle, engine: str) -> None:
    setter = getattr(io_handle, "set_engine", None)
    if callable(setter):
        setter(engine)
        return
    setter = getattr(io_handle, "SetEngine", None)
    if callable(setter):
        setter(engine)
        return
    raise AttributeError("Unable to set engine on ADIOS IO handle (missing set_engine/SetEngine method).")


def _optional_float(value: str) -> Optional[float]:
    lowered = value.strip().lower()
    if lowered in {"none", "auto"}:
        return None
    try:
        return float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"Expected a float or 'none', got '{value}'.") from exc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--instream", "-i", required=True, help="Input stream or BP file to read (e.g. results_.../output_3D).")
    parser.add_argument("--config", "-c", default="adios2.xml", help="Path to ADIOS2 configuration file (default: %(default)s).")
    parser.add_argument("--io-name", default="Output", help="Name of the IO object in the config (default: %(default)s).")
    parser.add_argument("--outdir", default="frames", help="Directory to store generated images (default: %(default)s).")
    parser.add_argument("--prefix", default="frame", help="Filename prefix for saved images (default: %(default)s).")
    parser.add_argument("--plane", choices=["xy", "xz", "yz", "all"], default="xy", help="Plane(s) to export (default: %(default)s).")
    parser.add_argument(
        "--plane-index",
        action="append",
        default=[],
        metavar="PLANE=VALUE",
        help="Override slice index (supports %%, fractions, 'min', 'mid', 'max').",
    )
    parser.add_argument(
        "--varname",
        "-v",
        default="velocity",
        help="Variable to read, e.g. 'velocity' or 'lbm_density' (default: %(default)s).",
    )
    parser.add_argument(
        "--component",
        choices=["x", "y", "z", "magnitude"],
        default="magnitude",
        help="Component for vector variables (default: %(default)s).",
    )
    parser.add_argument("--colormap", default="coolwarm", help="Matplotlib colour-map (default: %(default)s).")
    parser.add_argument("--figsize", default="10x4", help="Figure size in inches as WIDTHxHEIGHT (default: %(default)s).")
    parser.add_argument("--dpi", type=int, default=120, help="Output DPI for PNG files (default: %(default)s).")
    parser.add_argument("--vmin", type=float, default=-0.02, help="Lower colour scale bound (default: %(default)s).")
    parser.add_argument("--vmax", type=float, default=0.09, help="Upper colour scale bound (default: %(default)s).")
    parser.add_argument(
        "--autoscale-colour",
        "--autoscale-color",
        action="store_true",
        dest="autoscale_colour",
        help="Override --vmin/--vmax and let Matplotlib pick colour limits per frame.",
    )
    parser.add_argument("--max-frames", type=int, default=None, help="Limit the total number of exported steps (optional).")
    parser.add_argument("--nompi", "-nompi", action="store_true", help="Force serial ADIOS initialisation.")
    parser.add_argument("--log-level", default="INFO", help="Logging verbosity (default: %(default)s).")
    parser.add_argument(
        "--no-retry",
        action="store_true",
        help="Exit once the stream ends instead of waiting for it to reappear.",
    )
    parser.add_argument(
        "--retry-delay",
        type=float,
        default=2.0,
        help="Seconds to wait before trying to reopen the stream (default: %(default)s).",
    )
    return parser.parse_args()


def ensure_outdir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def export_frame(
    plt,
    gridspec,
    args: argparse.Namespace,
    plane: str,
    data: np.ndarray,
    selection_index: int,
    step: int,
    component_label: Optional[str],
    figsize: Tuple[float, float],
    outdir: Path,
    logger: logging.Logger,
) -> None:
    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(1, 1, figure=fig)
    ax = fig.add_subplot(gs[0, 0])

    if np.ma.isMaskedArray(data):
        data = np.asarray(data)
    cmap = plt.get_cmap(args.colormap)

    image = ax.imshow(
        data,
        origin="lower",
        interpolation="nearest",
        extent=[0, data.shape[1], 0, data.shape[0]],
        cmap=cmap,
        aspect="auto",
    )
    if args.vmin is not None or args.vmax is not None:
        image.set_clim(vmin=args.vmin, vmax=args.vmax)

    component_text = f" ({component_label})" if component_label else ""
    ax.set_title(f"{args.varname}{component_text} | {plane}  {selection_index} | step {step}")
    ax.set_xlabel(plane[0])
    ax.set_ylabel(plane[1])
    fig.tight_layout()

    filename = outdir / f"{args.prefix}_{plane}_{step:06d}.png"
    fig.savefig(filename, dpi=args.dpi, bbox_inches="tight")
    logger.info("Saved %s", filename)
    plt.close(fig)


def _consume_stream_once(
    io,
    args: argparse.Namespace,
    planes: Sequence[str],
    overrides: Mapping[str, str],
    mpi,
    plt,
    gridspec,
    figsize: Tuple[float, float],
    outdir: Path,
    logger: logging.Logger,
    state: Dict[str, Optional[Tuple[int, int, int]]],
    remaining_frames: Optional[int],
) -> int:
    reader = None
    frames_written = 0
    try:
        reader = Stream(io, args.instream, "r", mpi.comm)
        for fr_step in reader.steps():
            vars_info = fr_step.available_variables()
            var_present = args.varname in vars_info
            velocity_present = any(
                comp in vars_info for comp in (f"{args.varname}X", f"{args.varname}Y", f"{args.varname}Z")
            )
            if not var_present and not velocity_present:
                raise KeyError(f"Variable '{args.varname}' not available in the stream.")

            global_shape = state.get("global_shape")
            if global_shape is None:
                candidate_var = args.varname if var_present else f"{args.varname}X"
                shape_str_raw = vars_info[candidate_var]["Shape"]
                shape_str = shape_str_raw.replace("{", "").replace("}", "")
                dims = [int(dim.strip()) for dim in shape_str.split(",") if dim.strip()]
                global_shape = tuple(dims)  # type: ignore[assignment]
                if len(global_shape) != 3:
                    raise RuntimeError(f"Unsupported variable dimensionality {global_shape}.")
                state["global_shape"] = global_shape
                if mpi.rank == 0:
                    logger.info("Global lattice size (z, y, x): %s", global_shape)

            plane_indices: Dict[str, int] = {}
            for plane in planes:
                axis = PLANE_TO_AXIS[plane]
                max_index = global_shape[axis] - 1
                if plane in overrides:
                    try:
                        candidate = interpret_index(overrides[plane], max_index)
                    except ValueError as exc:
                        logger.warning("%s Using default mid-plane.", exc)
                        candidate = max_index // 2
                else:
                    candidate = max_index // 2
                plane_indices[plane] = clamp(candidate, 0, max_index)

            for plane in planes:
                selection = compute_plane_selection(plane, plane_indices[plane], global_shape)

                if var_present:
                    data = read_scalar(fr_step, args.varname, selection)
                else:
                    data = read_velocity(fr_step, args.varname, args.component, selection, vars_info)

                component_label = None if var_present else args.component

                if mpi.rank == 0:
                    export_frame(
                        plt,
                        gridspec,
                        args,
                        plane,
                        data,
                        plane_indices[plane],
                        fr_step.current_step(),
                        component_label,
                        figsize,
                        outdir,
                        logger,
                    )

            frames_written += 1
            if remaining_frames is not None and frames_written >= remaining_frames:
                return frames_written
    except KeyboardInterrupt:
        raise
    except Exception as exc:  # pragma: no cover - defensive logging for ADIOS issues
        logger.warning("Stream ended with %s", exc)
    finally:
        if reader is not None:
            reader.close()

    return frames_written


def main() -> None:
    args = parse_args()
    if getattr(args, "autoscale_colour", False):
        args.vmin = None
        args.vmax = None
    configure_logging(args.log_level)
    logger = logging.getLogger("tnl_lbm_export_frames")

    figsize = parse_figsize(args.figsize)
    outdir = Path(args.outdir)
    ensure_outdir(outdir)

    mpi = setup_mpi(args.nompi)
    if mpi.size > 1 and mpi.rank == 0:
        logger.info("Running in MPI mode with %d ranks (only rank 0 saves images).", mpi.size)

    plt, gridspec = load_matplotlib()

    adios = Adios(args.config, mpi.comm) if mpi.comm is not None else Adios(args.config)
    io = adios.declare_io(args.io_name)
    if not io_in_config_file(io):  # pragma: no cover - depends on config
        logger.warning("IO '%s' not present in %s; falling back to BP5 engine.", args.io_name, args.config)
        io_set_engine(io, "BP5")

    planes = expand_planes(args.plane)
    overrides = parse_plane_overrides(args.plane_index, logger)

    should_retry = not args.no_retry
    remaining_frames = args.max_frames
    state: Dict[str, Optional[Tuple[int, int, int]]] = {"global_shape": None}

    try:
        while True:
            frames = _consume_stream_once(
                io,
                args,
                planes,
                overrides,
                mpi,
                plt,
                gridspec,
                figsize,
                outdir,
                logger,
                state,
                remaining_frames,
            )

            if remaining_frames is not None:
                remaining_frames -= frames
                if remaining_frames is not None and remaining_frames <= 0:
                    logger.info("Reached max frame limit (%d). Stopping.", args.max_frames)
                    break

            if not should_retry:
                break

            if frames == 0:
                logger.debug("No frames read; waiting %.1fs before retrying.", args.retry_delay)
                state["global_shape"] = None
            else:
                logger.info("Stream ended after %d frame(s); waiting for new data (Ctrl+C to exit).", frames)

            time.sleep(max(args.retry_delay, 0.1))
    except KeyboardInterrupt:  # pragma: no cover
        logger.info("Interrupted by user; exiting.")


if __name__ == "__main__":
    main()
