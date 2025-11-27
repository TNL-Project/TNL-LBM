import argparse
import os
from paraview.simple import *
from paraview import print_info
from paraview import catalyst

# Catalyst options
options = catalyst.Options()
options.GlobalTrigger = "TimeStep"
options.EnableCatalystLive = 1
options.CatalystLiveTrigger = "TimeStep"

DISABLE_EXTRACTOR = True

if not DISABLE_EXTRACTOR:
    _project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    _extracts_dir = os.path.join(_project_root, "catalyst_output")
    os.makedirs(_extracts_dir, exist_ok=True)
    options.ExtractsOutputDirectory = _extracts_dir
    print_info("Catalyst extracts output directory: %s", options.ExtractsOutputDirectory)

# Setup render view
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

# Catalyst producer for inline visualization
def SetupCatalystProducer():
    producer = TrivialProducer(registrationName="fides")
    return producer

# Fides reader for post-hoc visualization
def SetupFidesReader(json, bp, sst):
    if json is None:
        fides = FidesReader(StreamSteps=1, FileName=bp)
        return fides
    
    fides = FidesJSONReader(StreamSteps=1, FileName=json)
    if sst:
        fides.DataSourceEngines = ["source", "SST"]
    fides.DataSourcePath = ["source", bp]
    fides.UpdatePipelineInformation()
    return fides

# Visualization pipeline
def SetupVisPipeline(producer, view):
    # Show producer
    producerDisplay = Show(producer, view, 'UniformGridRepresentation')
    view.ResetCamera()
    
    # Create velocity magnitude calculator
    calculator1 = Calculator(registrationName="VelocityMagnitude", Input=producer)
    calculator1.ResultArrayName = "velocity_magnitude"
    calculator1.Function = "sqrt(velocityX*velocityX + velocityY*velocityY + velocityZ*velocityZ)"
    
    # Create velocity vector
    calculator2 = Calculator(registrationName="VelocityVector", Input=calculator1)
    calculator2.ResultArrayName = "Velocity"
    calculator2.Function = "velocityX*iHat + velocityY*jHat + velocityZ*kHat"
    
    # Color transfer functions
    velocityMagLUT = GetColorTransferFunction("velocity_magnitude")
    velocityMagLUT.AutomaticRescaleRangeMode = "Clamp and update every timestep"
    velocityMagLUT.RescaleOnVisibilityChange = 1
    velocityMagLUT.ApplyPreset('Rainbow Desaturated', True)
    
    densityLUT = GetColorTransferFunction("lbm_density")
    densityLUT.AutomaticRescaleRangeMode = "Clamp and update every timestep"
    densityLUT.RescaleOnVisibilityChange = 1
    densityLUT.ApplyPreset('Cool to Warm', True)
    
    # Create slice
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
    
    # Show walls
    threshold1 = Threshold(registrationName="Walls", Input=producer)
    threshold1.Scalars = ["POINTS", "wall"]
    threshold1.LowerThreshold = 0.5
    threshold1.UpperThreshold = 10.0
    
    threshold1Display = Show(threshold1, view, "UniformGridRepresentation")
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
    
    return slice1, slice1Display


def _set_slice_origin_to_center(source, slice_filter):
    try:
        b = source.GetDataInformation().GetBounds()
        if b is None or len(b) != 6:
            return
        origin = [(b[0] + b[1]) / 2.0, (b[2] + b[3]) / 2.0, (b[4] + b[5]) / 2.0]
        slice_filter.SliceType.Origin = origin
    except Exception:
        # keep default origin if bounds are not available yet
        pass

# Setup extractor for PNG output
def SetupExtractor(view):
    pNG1 = CreateExtractor("PNG", view, registrationName="PNG1")
    pNG1.Trigger = "TimeStep"
    pNG1.Writer.FileName = "lbm_frame_{timestep:06d}.png"
    pNG1.Writer.ImageResolution = [1024, 768]
    pNG1.Writer.Format = "PNG"

# Catalyst execution callback
def catalyst_execute(info):
    print_info("in '%s::catalyst_execute'", __name__)
    global pipeline, display
    _set_slice_origin_to_center(producer, pipeline)
    pipeline.UpdatePipeline()
    
    print_info("executing (cycle={}, time={})".format(info.cycle, info.time))
    if producer.PointData["lbm_density"] is not None:
        print_info("Density range: {}".format(producer.PointData["lbm_density"].GetRange(0)))
    if producer.PointData["velocityX"] is not None:
        print_info("VelocityX range: {}".format(producer.PointData["velocityX"].GetRange(0)))

# Parse arguments
def ParseArgs():
    parser = argparse.ArgumentParser()
    parser.add_argument("-j", "--json_filename", help="path to Fides JSON file", type=str, required=False)
    parser.add_argument("-b", "--bp_filename", help="path to bp file", type=str, required=True)
    parser.add_argument("--staging", help="use SST engine", action="store_true")
    args = parser.parse_args()
    return args

# Streaming visualization (post-hoc)
def StreamingVis(args):
    NotReady = 1
    EndOfStream = 2
    
    fides = SetupFidesReader(args.json_filename, args.bp_filename, args.staging)
    view = SetupRenderView()
    
    step = 0
    while True:
        status = NotReady
        while status == NotReady:
            fides.PrepareNextStep()
            fides.UpdatePipelineInformation()
            status = fides.NextStepStatus
        if status == EndOfStream:
            return
        if step == 0:
            pipeline, display = SetupVisPipeline(fides, view)
        
        _set_slice_origin_to_center(fides, pipeline)
        pipeline.UpdatePipeline()
        output = f"lbm_output-{step:05d}.png"
        SaveScreenshot(output, view, ImageResolution=[1024, 768])
        step += 1

# Main entry point
if __name__ == "__main__":
    print("Running in post-hoc mode")
    args = ParseArgs()
    StreamingVis(args)
else:
    # Running from Catalyst inline
    print("Running in Catalyst inline mode")
    view = SetupRenderView()
    producer = SetupCatalystProducer()
    pipeline, display = SetupVisPipeline(producer, view)
    
    if not DISABLE_EXTRACTOR:
        SetupExtractor(view)
