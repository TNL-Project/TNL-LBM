\page UsersGuide TNL-LBM Users' Guide

# Users' Guide

Welcome to the users' guide for **TNL-LBM**, a lattice Boltzmann module built
on top of the Template Numerical Library (TNL). The goal of this guide is to
show how to set up, compile, and run reproducible simulations and how to extend
TNL-LBM in your own projects.

The guide is organised into the following thematic blocks:

1. [Getting Started](#ug_GettingStarted) provides the recommended CMake presets,
   explains how examples are organised, and shows the commands used by the
   documentation build.
2. [Lid-Driven Cavity](#ug_LidDrivenCavity) walks through a 3D cavity benchmark
   solved with a single-relaxation-time (SRT) model, including compilation,
   execution, and visualisation.
3. [Collision Operators](#ug_CollisionOperators) extends the cavity example to
   more advanced collision models such as MRT and CLBM and sums up best
   practices when switching between operators.
4. [Streaming Schemes](#ug_StreamingSchemes) compares the standard A-B and the
   A-A streaming patterns and highlights what needs to change in user code.
5. [Multi-GPU Execution](#ug_MultiGPU) demonstrates the minimal code changes
   required to exploit several MPI ranks and GPUs.

Each section accompanies runnable examples located under
`Documentation/UsersGuide`. The examples are compiled as part of the standard
CMake workflow and can be executed directly or via the helper targets defined in
this repository.

