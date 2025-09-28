# Streaming Schemes {#ug_StreamingSchemes}

[TOC]

## Overview

TNL-LBM implements both the orthodox A-B streaming (two distribution buffers)
and the memory-saving A-A streaming (single distribution buffer with pointer
swapping). The choice affects both performance and the boundary handling code.
This section shows how to reuse the driven cavity example with either scheme.

## A-B streaming (default)

When neither `AA_PATTERN` nor `AB_PATTERN` is defined, the library defaults to
A-B streaming. Two distribution buffers are allocated and the solver alternates
between them every iteration. The `runLidDrivenCavity` helper automatically
initialises the guard layers at index `0` and `N-1` in each dimension. These
layers are essential for the MPI synchronisation as well as for the bounce-back
boundaries configured in `setupBoundaries`.

## Switching to A-A streaming

The documentation target `lid_driven_cavity_aa` is built with the
`AA_PATTERN` compile definition enabled (see `Documentation/UsersGuide/StreamingSchemes/CMakeLists.txt`).
No source-code changes are required besides selecting the alternative target:

\snippet Documentation/UsersGuide/StreamingSchemes/lid_driven_cavity_aa.cu doc-lid-aa-run-start

The helper state keeps the extra guard layer initialised as `GEO_NOTHING`
explicitly to avoid race conditions during the even time steps of the A-A
pattern. When writing your own states, make sure to replicate this setup or to
extend the guard region according to your streaming stencil.

## Comparing schemes

A quick sanity check is to run both executables in documentation mode:

```
$ ./Documentation/UsersGuide/LidDrivenCavity/lid_driven_cavity_srt --doc-mode
$ ./Documentation/UsersGuide/StreamingSchemes/lid_driven_cavity_aa --doc-mode
```

Both should converge to the same centre-line velocity (within floating-point
noise). The A-A version allocates half the distribution memory but performs more
arithmetic per time step. Profiling on your hardware helps deciding which
variant is preferable for a particular workload.

