#!/usr/bin/env python3
"""End-to-end ParaView check for the 3-level nesting mock VTKHDF output.

Nesting arm of tests/test_amr_paraview_e2e.py (the amr-nlevel-nesting
plan's commit D): tests/test_amr_vtkhdf_writer.cu pins the writer's HDF5
layout on a 3-level chain; this script proves the nested-region file works
for its real consumer. The input is produced by the dedicated driver
tests/test_amr_nesting_sim (a 32^3 periodic Taylor-Green box with three
fine blocks telescoping [6,26) -> [15,25) -> [33,41) per parent frame) --
sim_AMR/sim_AMR_channel hardcode their region specs, so the e2e arm drives
the mock instead.

Run under pvpython (ParaView's bundled python), e.g. via the wrapper
tests/test_amr_paraview_e2e_nesting.sh, or directly from the project root:

    pvpython tests/test_amr_paraview_e2e_nesting.py \
        --input results_test_amr_nesting_sim_np001/output_amr_0000.vtkhdf \
        --outdir /tmp/opencode/pv_e2e_nesting

One PASS/FAIL line per check (same style as the C++ report() helper); prints
"RESULT: all ParaView E2E nesting checks passed" and exits 0 on success,
exits 1 if any check failed.
"""

import argparse
import os
import sys

import numpy as np

import paraview.simple as pvs
from paraview import servermanager

# use vtkmodules.* directly: the vtk.py umbrella import pulls in the OpenXR
# module, whose shared library may be missing on headless machines
from vtkmodules.numpy_interface import dataset_adapter as dsa

EXPECTED_FIELDS = ("rho", "vx", "vy", "vz", "vtkGhostType")
# 4 AMR levels (0..3), one block per level
EXPECTED_LEVELS = 4
EXPECTED_BLOCKS_PER_LEVEL = 1
# per-level block cell counts (the writer emits the interior plus the
# footprint-covering ghost rows): L0 32^3, L1 40^3, L2 20^3, L3 16^3
EXPECTED_CELLS = {0: 32**3, 1: 40**3, 2: 20**3, 3: 16**3}
# per-level REFINEDCELL censuses: L0 <- level-1 footprint [6,26)^3 = 20^3,
# L1 <- level-2 footprint [15,25)^3 = 10^3, L2 <- level-3 footprint
# [33,41)^3 = 8^3, L3 never (finest)
EXPECTED_REFINED_CELLS = {0: 20**3, 1: 10**3, 2: 8**3, 3: 0}
# rho sanity bound: same Taylor-Green amplitude class as sim_AMR at res 1
# (max envelope 9*V0^2/8 ~= 6.9e-5), the 5e-4 bound covers the IC envelope
# plus early-run evolution/interface residual on the nested path
RHO_TOLERANCE = 5e-4
VX_ABS_MAX = 0.02
# vtkDataSetAttributes::REFINEDCELL; the reader ORs in EXTERIORCELL (0x8)
# when it blanks overlapped cells, so the loaded array contains 4|8=12,
# not plain 4
REFINEDCELL_BIT = 0x4
EXTERIORCELL_BIT = 0x8
MIN_PNG_BYTES = 30 * 1024
MIN_UNIQUE_COLORS = 64

g_failures = 0


def report(ok, what, actual=None):
    global g_failures
    if ok:
        print(f"PASS: {what}", flush=True)
    else:
        suffix = f" (actual: {actual})" if actual is not None else ""
        print(f"FAIL: {what}{suffix}", flush=True)
        g_failures += 1


def check_structure(src):
    report(
        type(src).__name__ == "VTKHDFReader",
        "OpenDataFile uses the VTKHDF reader",
        type(src).__name__,
    )
    # reader delivers a vtkOverlappingAMR composite to the client
    oamr = src.GetClientSideObject().GetOutputDataObject(0)
    report(
        oamr.GetClassName() == "vtkOverlappingAMR",
        "loaded data object is a vtkOverlappingAMR",
        oamr.GetClassName(),
    )
    n_levels = oamr.GetNumberOfLevels()
    report(n_levels == EXPECTED_LEVELS, "AMR has 4 levels", n_levels)
    for level in range(min(n_levels, EXPECTED_LEVELS)):
        n_blocks = oamr.GetNumberOfBlocks(level)
        report(
            n_blocks == EXPECTED_BLOCKS_PER_LEVEL,
            f"level {level} has 1 block",
            n_blocks,
        )
    fields = set()
    for i in range(src.CellData.GetNumberOfArrays()):
        fields.add(src.CellData.GetArray(i).Name)
    for name in EXPECTED_FIELDS:
        report(name in fields, f"cell data field '{name}' exists", sorted(fields))
    return oamr


def check_values(oamr):
    for level in range(EXPECTED_LEVELS):
        block = oamr.GetDataSet(level, 0)
        if block is None:
            report(False, f"level {level} block 0 readable", "GetDataSet returned None")
            continue
        data = dsa.WrapDataObject(block)
        n_cells = block.GetNumberOfCells()
        report(
            n_cells == EXPECTED_CELLS[level],
            f"level {level} cell count is {EXPECTED_CELLS[level]}",
            n_cells,
        )
        rho = np.asarray(data.CellData["rho"])
        ghost = np.asarray(data.CellData["vtkGhostType"])
        # rho finite everywhere (the coupling must not emit NaN/garbage),
        # then the TG-envelope sanity bound on the visible cells
        report(
            bool(np.isfinite(rho).all()),
            f"level {level} rho finite on all emitted cells",
            f"{int(np.count_nonzero(~np.isfinite(rho)))} non-finite cells",
        )
        # skip reader-hidden cells (blanked coarse interior, hidden fine ghost)
        visible_mask = (ghost & EXTERIORCELL_BIT) == 0
        rho_err = float(np.abs(rho[visible_mask] - 1.0).max())
        report(
            rho_err <= RHO_TOLERANCE,
            f"level {level} rho within 1.0 +/- {RHO_TOLERANCE} (TG envelope)",
            f"max deviation {rho_err:.3e}",
        )
        if level == 0:
            vx = np.asarray(data.CellData["vx"])
            vx_max = float(np.abs(vx).max())
            report(
                vx_max < VX_ABS_MAX,
                f"level 0 |vx| < {VX_ABS_MAX}",
                f"max {vx_max:.4f}",
            )
        n_refined = int(np.count_nonzero((ghost & REFINEDCELL_BIT) == REFINEDCELL_BIT))
        report(
            n_refined == EXPECTED_REFINED_CELLS[level],
            f"level {level} REFINEDCELL count is {EXPECTED_REFINED_CELLS[level]}",
            n_refined,
        )
        # the reader's overlap-blanking must not exceed the writer's
        # REFINEDCELL footprint at ANY level (an EXTERIORCELL-only cell --
        # 0x8 without 0x4 -- is the interface-ring blanking defect class of
        # the 2-level idiom, generalized per level)
        n_blank_only = int(
            np.count_nonzero(
                ((ghost & EXTERIORCELL_BIT) != 0) & ((ghost & REFINEDCELL_BIT) == 0)
            )
        )
        report(
            n_blank_only == 0,
            f"level {level} has no EXTERIORCELL-only cells (blanking == REFINEDCELL)",
            n_blank_only,
        )
        if level == EXPECTED_LEVELS - 1:
            valid_tags = set(np.unique(ghost).tolist()) <= {0}
            report(
                valid_tags,
                f"level {level} vtkGhostType is 0 everywhere (finest level)",
                np.unique(ghost).tolist(),
            )


def check_fetch(src):
    # Fetch proves client-side delivery; note that on OverlappingAMR the
    # fetched copy flattens the level tree (0 levels in ParaView 6.1), so the
    # per-level value checks above use the pipeline output object instead
    fetched = servermanager.Fetch(src)
    report(
        fetched is not None and fetched.GetClassName() == "vtkOverlappingAMR",
        "servermanager.Fetch delivers a vtkOverlappingAMR to the client",
        None if fetched is None else fetched.GetClassName(),
    )


def check_render(src, outdir):
    os.makedirs(outdir, exist_ok=True)
    png_path = os.path.join(outdir, "amr_nesting_slice.png")
    bounds = src.GetDataInformation().GetBounds()
    center = [
        0.5 * (bounds[0] + bounds[1]),
        0.5 * (bounds[2] + bounds[3]),
        0.5 * (bounds[4] + bounds[5]),
    ]

    view = pvs.CreateRenderView()
    view.ViewSize = [1024, 800]

    slice_filter = pvs.Slice(Input=src)
    slice_filter.SliceType = "Plane"
    slice_filter.SliceType.Origin = center
    slice_filter.SliceType.Normal = [0.0, 0.0, 1.0]
    slice_filter.UpdatePipeline()

    display = pvs.Show(slice_filter, view)
    pvs.ColorBy(display, ("CELLS", "rho"))
    # rho sits on 1.0 within a few ULPs; stretch the lookup table across that
    # tiny range so the cells shade differently and the render carries real
    # signal (a near-blank image would compress to a few KB)
    lut = pvs.GetColorTransferFunction("rho")
    lut.RescaleTransferFunction(1.0 - 1e-5, 1.0 + 1e-5)
    lut.ApplyPreset("Cool to Warm", True)
    display.SetRepresentationType("Surface With Edges")
    display.Ambient = 0.3
    display.Diffuse = 0.9
    scalar_bar = pvs.GetScalarBar(lut, view)
    scalar_bar.Visibility = 1
    pvs.Hide(src, view)  # show the slice only

    # angled 3/4 view so lighting shades the cells (richer, verifiable image)
    view.CameraPosition = [center[0] + 0.55, center[1] - 0.75, center[2] + 0.65]
    view.CameraFocalPoint = center
    view.CameraViewUp = [0.0, 0.0, 1.0]
    pvs.SaveScreenshot(
        png_path, view, ImageResolution=[1024, 800], TransparentBackground=0
    )
    # tear down the render view before interpreter shutdown: destroying the
    # offscreen render window during pvpython's atexit teardown segfaults
    pvs.Delete(view)

    exists = os.path.isfile(png_path)
    report(exists, "slice screenshot written", png_path)
    if not exists:
        return
    size = os.path.getsize(png_path)
    report(
        size > MIN_PNG_BYTES,
        f"screenshot larger than {MIN_PNG_BYTES // 1024} KB",
        f"{size} bytes",
    )
    try:
        from PIL import Image
    except ImportError:
        print(
            "SKIP: PIL not available; size/extension check stands in for pixel content",
            flush=True,
        )
        return
    with Image.open(png_path) as image:
        colors = image.convert("RGB").getcolors(maxcolors=4_000_000)
    unique = (
        len(colors) if colors is not None else MIN_UNIQUE_COLORS
    )  # saturated => plenty
    report(
        unique >= MIN_UNIQUE_COLORS,
        f"screenshot has at least {MIN_UNIQUE_COLORS} unique colors",
        unique,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", default="results_test_amr_nesting_sim_np001/output_amr_0000.vtkhdf"
    )
    parser.add_argument("--outdir", default="/tmp/opencode/pv_e2e_nesting")
    args = parser.parse_args()

    input_ok = os.path.isfile(args.input)
    report(input_ok, "input VTKHDF file exists", args.input)
    if not input_ok:
        finish()

    try:
        src = pvs.OpenDataFile(os.path.abspath(args.input))
        src.UpdatePipeline()
        oamr = check_structure(src)
        check_fetch(src)
        check_values(oamr)
        check_render(src, args.outdir)
    except Exception as exc:
        report(
            False,
            "ParaView pipeline completed without exceptions",
            f"{type(exc).__name__}: {exc}",
        )
    finish()


def finish():
    if g_failures == 0:
        print("RESULT: all ParaView E2E nesting checks passed", flush=True)
        sys.exit(0)
    print(f"RESULT: {g_failures} ParaView E2E nesting check(s) FAILED", flush=True)
    sys.exit(1)


if __name__ == "__main__":
    main()
