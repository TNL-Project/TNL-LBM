import os
from paraview.simple import *
from paraview import print_info
from paraview import catalyst

# Catalyst options
options = catalyst.Options()
options.GlobalTrigger = "TimeStep"
options.EnableCatalystLive = 1
options.CatalystLiveTrigger = "TimeStep"


# ---------------------------------------------------------------------------
#  Render view
# ---------------------------------------------------------------------------
def SetupRenderView():
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


def SetupCatalystProducer():
    producer = TrivialProducer(registrationName="fides")
    return producer


# ---------------------------------------------------------------------------
#  Visualization pipeline
# ---------------------------------------------------------------------------

def SetupVisPipeline(producer, view):
    producerDisplay = Show(producer, view, 'GeometryRepresentation')
    view.ResetCamera()

    # Velocity magnitude & vector
    calculator1 = Calculator(registrationName="VelocityMagnitude", Input=producer)
    calculator1.ResultArrayName = "velocity_magnitude"
    calculator1.Function = "sqrt(velocity_x*velocity_x + velocity_y*velocity_y + velocity_z*velocity_z)"

    calculator2 = Calculator(registrationName="VelocityVector", Input=calculator1)
    calculator2.ResultArrayName = "Velocity"
    calculator2.Function = "velocity_x*iHat + velocity_y*jHat + velocity_z*kHat"

    # Color maps
    velocityMagLUT = GetColorTransferFunction("velocity_magnitude")
    velocityMagLUT.AutomaticRescaleRangeMode = "Clamp and update every timestep"
    velocityMagLUT.RescaleOnVisibilityChange = 1
    velocityMagLUT.ApplyPreset('Rainbow Desaturated', True)
    
    densityLUT = GetColorTransferFunction("lbm_density")
    densityLUT.AutomaticRescaleRangeMode = "Clamp and update every timestep"
    densityLUT.RescaleOnVisibilityChange = 1
    densityLUT.ApplyPreset('Cool to Warm', True)

    # Slice coloured by velocity magnitude
    slice1 = Slice(registrationName="Slice1", Input=calculator2)
    slice1.SliceType = "Plane"
    slice1.SliceOffsetValues = [0.0]
    slice1.SliceType.Normal = [0.0, 0.0, 1.0]

    slice1Display = Show(slice1, view, "GeometryRepresentation")
    slice1Display.Representation = "Surface"
    slice1Display.ColorArrayName = ["POINTS", "velocity_magnitude"]
    slice1Display.LookupTable = velocityMagLUT
    
    # Create contours for density
    contour1 = Contour(registrationName="DensityContour", Input=producer)
    contour1.ContourBy = ["POINTS", "lbm_density"]
    contour1.Isosurfaces = [1.0]
    contour1.PointMergeMethod = "Uniform Binning"
    
    contour1Display = Show(contour1, view, "GeometryRepresentation")
    contour1Display.Representation = "Surface"
    contour1Display.ColorArrayName = ["POINTS", "lbm_density"]
    contour1Display.LookupTable = densityLUT
    contour1Display.Opacity = 0.5

    # Walls
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
    
    # Color bar
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


def _set_slice_origin_to_center(source, slice_filter):
    """Move slice plane to the centre of the dataset bounding box."""
    try:
        b = source.GetDataInformation().GetBounds()
        if b is None or len(b) != 6:
            return
        origin = [(b[0] + b[1]) / 2.0, (b[2] + b[3]) / 2.0, (b[4] + b[5]) / 2.0]
        slice_filter.SliceType.Origin = origin
    except Exception:
        pass


# ---------------------------------------------------------------------------
#  Catalyst execution callback
# ---------------------------------------------------------------------------

pipeline_filters = None

def catalyst_execute(info):
    global pipeline_filters

    if pipeline_filters is None:
        pipeline_filters = SetupVisPipeline(producer, view)

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


# ---------------------------------------------------------------------------
#  Initialize for Catalyst inline mode
# ---------------------------------------------------------------------------
view = SetupRenderView()
producer = SetupCatalystProducer()
