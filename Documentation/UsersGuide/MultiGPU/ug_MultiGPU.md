# Multi-GPU Execution {#ug_MultiGPU}

[TOC]

## Background

TNL-LBM relies on MPI to distribute subdomains across ranks. When the code is
compiled with CUDA support each rank can be mapped to its own GPU. The
constructor of `LBM` automatically partitions the global lattice using a block
wise decomposition, so the only missing piece is the GPU-selection policy.

## Selecting a GPU per rank

The example `lid_driven_cavity_multi_gpu` augments the base solver with a small
helper that assigns a CUDA device to each rank. The mapping can be automatic or
explicitly controlled through `--local-device`:

\snippet Documentation/UsersGuide/MultiGPU/lid_driven_cavity_multi_gpu.cu doc-mgpu-select-start

All other aspects reuse the generic runner discussed in the previous sections:

\snippet Documentation/UsersGuide/MultiGPU/lid_driven_cavity_multi_gpu.cu doc-mgpu-run-start

## Launching with `mpirun`

Assuming two GPUs are accessible on the host, the following command starts a
2-way run while leaving the device mapping to the helper above:

```
$ mpirun -np 2 ./Documentation/UsersGuide/MultiGPU/lid_driven_cavity_multi_gpu \
    --resolution 2 --final-time 5e-3 --vtk-period -1
```

Set `CUDA_VISIBLE_DEVICES` beforehand if you want to control which physical
GPUs participate in the run. Alternatively, specify a concrete GPU id per rank
(`--local-device 1` on the second rank, for example) to override the round-robin
assignment.

## Post-processing distributed results

Each rank writes its own `results_<id>/vtk3D/rankXXX_*.vtk` files. ParaView can
load the entire set via the **Open File** dialogue by selecting all ranks at
once. For scripted workflows, concatenate the partitions using the
`vtkAppendFilter` or leverage the ADIOS2 output available in the core framework
for large-scale studies.

