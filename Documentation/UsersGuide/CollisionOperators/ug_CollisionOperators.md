# Alternative Collision Operators {#ug_CollisionOperators}

[TOC]

## Motivation

The single-relaxation-time (SRT) model is simple and efficient but may suffer
from accuracy and stability limitations at higher Reynolds numbers. TNL-LBM
ships with a number of alternative collision operators. This section illustrates
how to switch the driven-cavity example to the multiple-relaxation-time (MRT)
and the cumulant-based (CLBM) formulations.

## Switching to MRT

The MRT variant dampens numerical artefacts by relaxing the moments of the
particle distribution at different rates. The driver below reuses the common
lattice setup code and only substitutes the collision class:

\snippet Documentation/UsersGuide/CollisionOperators/lid_driven_cavity_mrt.cu doc-lid-mrt-run-start

Usage mirrors the SRT solver. The same command-line options configure the
lattice and the physical parameters, making it straightforward to compare the
results:

```
$ ./Documentation/UsersGuide/CollisionOperators/lid_driven_cavity_mrt --doc-mode
```

Monitor the stability threshold by gradually increasing the Reynolds number.
For MRT the lattice viscosity (`--lbm-viscosity`) still sets the baseline
relaxation time, but the model adapts the effective relaxation parameter using a
Smagorinsky-like formulation inside the collision routine.

## Switching to CLBM

The central-moment formulation (CLBM) improves Galilean invariance and handles
shear flows with lower numerical diffusion. Enabling it requires a single-line
change:

\snippet Documentation/UsersGuide/CollisionOperators/lid_driven_cavity_clbm.cu doc-lid-clbm-run-start

The remaining code is identical to the SRT and MRT variants, so the output
folders follow exactly the same naming convention. When comparing collision
operators, store the results under separate suffixes by passing
`--id-suffix _mrt` or similar to simplify post-processing.

## Practical tips

- All collision operators share the same `LidDrivenCavityOptions` structure and
  honour the `--doc-mode` flag. Switching models therefore does not affect the
  automation scripts that sweep physical parameters.
- If you customise collision parameters (e.g. relaxation rates for MRT), prefer
  wrapping them inside a new options structure so that the solver's public API
  remains backwards compatible.
- Always keep an eye on the density deviation reported by the solver. Large
  oscillations typically indicate that the chosen combination of lattice
  viscosity and resolution violates the stability bounds of the selected
  collision model.

