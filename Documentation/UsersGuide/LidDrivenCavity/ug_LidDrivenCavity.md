# Lid-Driven Cavity {#ug_LidDrivenCavity}

[TOC]

## Problem outline

The lid-driven cavity is a canonical benchmark for incompressible flow
simulations. A cubic cavity is filled with a Newtonian fluid. All faces are
stationary except the top lid that moves with a constant velocity in the
positive \f$x\f$ direction. The non-slip boundary condition leads to a
recirculating flow with a primary vortex at the cavity centre and secondary
vortices in the corners.

TNL-LBM models the problem with the D3Q27 lattice. The example in
`Documentation/UsersGuide/LidDrivenCavity/lid_driven_cavity_srt.cu` uses the
single-relaxation-time (BGK) collision operator and demonstrates a minimal
solver setup.

## Creating the lattice and boundary conditions

The helper header `lbm_doc_common.h` builds the lattice and sets up the
`State` subclass that represents our simulation:

\snippet Documentation/UsersGuide/common/lbm_doc_common.h doc-lid-lattice-start

The state class configures the moving-lid boundary by reusing the existing
`GEO_INFLOW_LEFT` condition provided by TNL-LBM. The lid speed is injected
through `block.data.inflow_vx` in `updateKernelVelocities`, while the remaining
faces are marked as solid walls:

\snippet Documentation/UsersGuide/common/lbm_doc_common.h doc-lid-state-start

## Running the SRT solver

The SRT executable wires the generic runner with the collision model:

\snippet Documentation/UsersGuide/LidDrivenCavity/lid_driven_cavity_srt.cu doc-lid-srt-run-start

A typical run at Reynolds number 100 (lid velocity `0.1 m/s`, cavity length
`0.01 m`, viscosity `1e-6 m^2/s`) looks as follows:

```
$ ./Documentation/UsersGuide/LidDrivenCavity/lid_driven_cavity_srt \
      --resolution 2 --lid-velocity 0.1 --length 0.01 --viscosity 1e-6 \
      --final-time 5e-3 --vtk-period 1e-3
Cavity length: 1.000e-02 m, lattice spacing: 1.562e-04 m
Time step: 4.069e-06 s, Reynolds number: 100.0
Velocity at cavity centre: (9.337e-02, -1.103e-03, -1.028e-03) m/s
```

The console output reports the physical lattice spacing, the time step
obtained from the lattice viscosity, and the instantaneous velocity measured at
cavity centre once the simulation finishes.

## Inspecting the results

If VTK exports are enabled, the solver writes 3D snapshots into
`results_lid_driven_cavity_srt_resXX/vtk3D`. Open the directory in ParaView, and
use the following filters to analyse the flow:

1. **Warp By Vector** with the `velocity` field to obtain a qualitative view of
   the deformation of the grid caused by the flow.
2. **Stream Tracer** seeded near the lid to visualise the recirculation.
3. **Calculator** to derive derived quantities (e.g. vorticity magnitude) from
   the exported velocity field.

For 2D slices, enable `--cut-period` and inspect the generated files under
`results_.../cuts`.

## Automating parameter studies

The command-line interface exposes all essential parameters. Tuning the
Reynolds number is as simple as changing `--lid-velocity` or `--viscosity`. The
helper struct parses the arguments and stores them in
`LidDrivenCavityOptions`:

\snippet Documentation/UsersGuide/common/lbm_doc_common.h doc-lid-options-start

Because the runner converts all physical parameters to lattice units in one
place, the rest of the code remains untouched when performing parameter
studies.

