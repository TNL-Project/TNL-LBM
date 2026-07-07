# ruff: noqa: F403, F405, ANN401
# pyright: reportWildcardImportFromLibrary=false, reportUndefinedVariable=false, reportOptionalMemberAccess=false, reportAttributeAccessIssue=false, reportCallIssue=false
"""Headless Catalyst pipeline for the TNL-LBM inline plugin test.

This pipeline keeps Catalyst Live enabled (so the ADIOS2 inline reader is not
used) and uses a PNG extractor to save each rendered frame to disk.  It is used
by tests/adios2-inline-plugin.xml for the output-data regression test.
"""

import os
from pathlib import Path
from typing import Any

from paraview import catalyst, print_info
from paraview.simple import *

_state: dict[str, Any] = {
    "step_count": 0,
    "pipeline_filters": None,
    "screenshot_dir": None,
}

options = catalyst.Options()
options.GlobalTrigger = "TimeStep"
options.EnableCatalystLive = 1
options.CatalystLiveTrigger = "TimeStep"


def SetupRenderView() -> Any:
    view = CreateView("RenderView")
    try:
        view.UseOffscreenRendering = 1
    except Exception:
        pass
    camera = GetActiveCamera()
    camera.Azimuth(45)
    camera.Elevation(30)

    SetActiveView(None)
    layout1 = CreateLayout(name="LBM Layout")
    layout1.AssignView(0, view)
    layout1.SetSize(1024, 768)

    SetActiveView(view)
    return view


def SetupCatalystProducer() -> Any:
    producer = TrivialProducer(registrationName="fides")
    return producer


def SetupVisPipeline(producer: Any, view: Any) -> dict[str, Any]:
    _ = Show(producer, view, "GeometryRepresentation")
    view.ResetCamera()

    calculator1 = Calculator(registrationName="VelocityMagnitude", Input=producer)
    calculator1.ResultArrayName = "velocity_magnitude"
    calculator1.Function = (
        "sqrt(velocity_x*velocity_x + velocity_y*velocity_y + velocity_z*velocity_z)"
    )

    calculator2 = Calculator(registrationName="VelocityVector", Input=calculator1)
    calculator2.ResultArrayName = "Velocity"
    calculator2.Function = "velocity_x*iHat + velocity_y*jHat + velocity_z*kHat"

    velocityMagLUT = GetColorTransferFunction("velocity_magnitude")
    velocityMagLUT.AutomaticRescaleRangeMode = "Clamp and update every timestep"
    velocityMagLUT.RescaleOnVisibilityChange = 1
    velocityMagLUT.ApplyPreset("Rainbow Desaturated", True)

    densityLUT = GetColorTransferFunction("lbm_density")
    densityLUT.AutomaticRescaleRangeMode = "Clamp and update every timestep"
    densityLUT.RescaleOnVisibilityChange = 1
    densityLUT.ApplyPreset("Cool to Warm", True)

    slice1 = Slice(registrationName="Slice1", Input=calculator2)
    slice1.SliceType = "Plane"
    slice1.SliceOffsetValues = [0.0]
    slice1.SliceType.Normal = [0.0, 0.0, 1.0]

    slice1Display = Show(slice1, view, "GeometryRepresentation")
    slice1Display.Representation = "Surface"
    slice1Display.ColorArrayName = ["POINTS", "velocity_magnitude"]
    slice1Display.LookupTable = velocityMagLUT

    contour1 = Contour(registrationName="DensityContour", Input=producer)
    contour1.ContourBy = ["POINTS", "lbm_density"]
    contour1.Isosurfaces = [1.0]
    contour1.PointMergeMethod = "Uniform Binning"

    contour1Display = Show(contour1, view, "GeometryRepresentation")
    contour1Display.Representation = "Surface"
    contour1Display.ColorArrayName = ["POINTS", "lbm_density"]
    contour1Display.LookupTable = densityLUT
    contour1Display.Opacity = 0.5

    threshold1 = Threshold(registrationName="Walls", Input=producer)
    threshold1.Scalars = ["POINTS", "wall"]
    threshold1.LowerThreshold = 0.5
    threshold1.UpperThreshold = 10.0

    threshold1Display = Show(threshold1, view, "GeometryRepresentation")
    threshold1Display.Representation = "Surface"
    threshold1Display.ColorArrayName = [None, ""]
    threshold1Display.AmbientColor = [0.5, 0.5, 0.5]
    threshold1Display.DiffuseColor = [0.5, 0.5, 0.5]
    threshold1Display.Opacity = 0.3

    Hide(producer, view)

    velocityMagColorBar = GetScalarBar(velocityMagLUT, view)
    velocityMagColorBar.Title = "Velocity Magnitude"
    velocityMagColorBar.ComponentTitle = ""
    slice1Display.SetScalarBarVisibility(view, True)

    return {
        "slice": slice1,
        "contour": contour1,
        "threshold": threshold1,
        "slice_display": slice1Display,
    }


def _set_slice_origin_to_center(source: Any, slice_filter: Any) -> None:
    try:
        b = source.GetDataInformation().GetBounds()
        if b is None or len(b) != 6:
            return
        origin = [(b[0] + b[1]) / 2.0, (b[2] + b[3]) / 2.0, (b[4] + b[5]) / 2.0]
        slice_filter.SliceType.Origin = origin
    except Exception:
        pass


def _get_screenshot_dir() -> Path:
    env_dir = os.environ.get("TNL_LBM_SCREENSHOT_DIR")
    if env_dir:
        return Path(env_dir).resolve()
    candidates = sorted(Path.cwd().glob("results_test_outputdata_*"))
    if candidates:
        return candidates[0] / "catalyst_screenshots"
    return Path.cwd() / "catalyst_screenshots"


def catalyst_initialize() -> None:
    _state["step_count"] = 0
    _state["pipeline_filters"] = None
    _state["screenshot_dir"] = None
    screenshot_dir = _get_screenshot_dir()
    screenshot_dir.mkdir(parents=True, exist_ok=True)
    png_extractor = CreateExtractor("PNG", view, registrationName="PNG1")
    png_extractor.Writer.FileName = str(screenshot_dir / "frame_{timestep:04d}.png")
    png_extractor.Writer.ImageResolution = [1024, 768]
    _state["screenshot_dir"] = screenshot_dir


def catalyst_execute(info: Any) -> None:
    _state["step_count"] += 1

    pipeline_filters = _state.get("pipeline_filters")
    if pipeline_filters is None:
        pipeline_filters = SetupVisPipeline(producer, view)
        _state["pipeline_filters"] = pipeline_filters

    view.ViewTime = info.time

    _set_slice_origin_to_center(producer, pipeline_filters["slice"])

    pipeline_filters["slice"].UpdatePipeline()
    pipeline_filters["contour"].UpdatePipeline()
    pipeline_filters["threshold"].UpdatePipeline()

    view.ResetCamera()

    try:
        Render(view)
    except Exception:
        pass

    print_info("catalyst_execute  cycle=%d  time=%.6f", info.cycle, info.time)


def catalyst_finalize() -> None:
    path = os.environ.get("TNL_LBM_CATALYST_STEPS_FILE")
    if path:
        with open(path, "w") as f:
            f.write(f"{_state['step_count']}\n")


view = SetupRenderView()
producer = SetupCatalystProducer()
