# Getting Started {#ug_GettingStarted}

[TOC]

## Prerequisites

TNL-LBM builds on top of the Template Numerical Library and therefore relies on
CMake (>= 3.28), a C++17 toolchain, and a CUDA capable GPU when running the
examples in this guide. A working MPI installation is recommended for the
multi-GPU section, but single-rank runs do not require it. The repository
already contains all third-party dependencies through CMake's
`FetchContent` directives, so an internet connection is only needed during the
first configuration.

## Configuring the project

The users' guide ships with a small set of documentation examples located under
`Documentation/UsersGuide`. They are enabled automatically after adding the
`Documentation` subdirectory in the top-level `CMakeLists.txt`. A typical build
sequence looks as follows:

```
$ cmake --preset release
$ cmake --build --preset release
```

The documentation targets piggyback on the library build and are therefore
compiled together with the rest of TNL-LBM. Once the project is built, each
example becomes available as an executable in the binary directory. For
instance, the SRT driven-cavity solver can be launched manually via

```
$ ./Documentation/UsersGuide/LidDrivenCavity/lid_driven_cavity_srt --doc-mode
```

The `--doc-mode` switch reproduces the configuration used by the documentation
build. It reduces the lattice size and simulation length so that the executable
finishes within a few seconds.

## Understanding the example skeleton

Every driven-cavity example follows the structure sketched below. The CLI layer
is shared across solvers and is set up by

\snippet Documentation/UsersGuide/LidDrivenCavity/lid_driven_cavity_srt.cu doc-lid-cli-parse-start

The heavy lifting happens inside a reusable runner defined in
`Documentation/UsersGuide/common/lbm_doc_common.h`:

\snippet Documentation/UsersGuide/common/lbm_doc_common.h doc-lid-runner-start

The runner prepares the lattice, instantiates the selected collision operator,
updates run-time counters, and executes the LBM time loop. Specialised examples
(such as MRT or CLBM) only choose a different collision type while reusing the
rest of the machinery.

## Aliasing documentation targets

For convenience, a single aggregate target invokes all documentation examples:

```
$ cmake --build --preset release --target run-doc-examples
```

Running this target populates the `output_snippets` directory inside the build
folder with the console output captured during each invocation. Doxygen includes
those excerpts directly in the guide.

## Visualising the output in ParaView

By default the users' guide disables VTK exports to keep the CI runs fast. To
produce visualisation data, pass a positive `--vtk-period` (in physical seconds)
when launching an example:

```
$ ./Documentation/UsersGuide/LidDrivenCavity/lid_driven_cavity_srt \
    --vtk-period 1e-3 --final-time 5e-3 --resolution 2
```

VTK files are written under `results_<run-id>/vtk3D`. Open the directory in
ParaView and apply the **Warp By Vector** and **Stream Tracer** filters to
explore the three-dimensional flow field. For scripted post-processing consider
using the VTK Python API or `PyVista`.

