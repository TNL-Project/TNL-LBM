# TNL-LBM

__TNL-LBM__ is an implementation of the __Lattice Boltzmann Method__ using the
[__Template Numerical Library__](https://gitlab.com/tnl-project/tnl).
This repository contains a general framework for writing LBM-based solvers and
a few simple examples that show how to adapt the code for a particular problem.

TNL-LBM is a high-performance lattice Boltzmann code for direct numerical
simulations (DNS) of turbulent flow. It was verified and validated on multiple
problems, see the publications in the [Citing](#citing) section. The main
features are:

- Modular architecture with pluggable components (collision operators,
  streaming patterns, boundary conditions, macroscopic quantities, etc).
    - [Cumulant collision operator][CuLBM] for D3Q27.
    - The [A-A pattern][A-A pattern] streaming scheme can be employed to
      significantly reduce memory consumption.
- Optimized data layout on uniform lattice based on the [NDArray][NDArray]
  data structure from TNL.
- Scalable distributed computing based on [CUDA-aware MPI][CUDA-aware MPI] and
  [DistributedNDArraySynchronizer][DistributedNDArraySynchronizer] from TNL.
    - Good parallel efficiency is ensured by overlapping computation and
      communication.

[CuLBM]: https://doi.org/10.1016/j.camwa.2015.05.001
[A-A pattern]: https://doi.org/10.1109/ICPP.2009.38
[NDArray]: https://tnl-project.gitlab.io/tnl/ug_NDArrays.html
[DistributedNDArraySynchronizer]: https://tnl-project.gitlab.io/tnl/classTNL_1_1Containers_1_1DistributedNDArraySynchronizer.html
[CUDA-aware MPI]: https://developer.nvidia.com/blog/introduction-cuda-aware-mpi/

## Getting started

1. Install [Git](https://git-scm.com/).

2. Clone the repository:

       git clone https://gitlab.com/tnl-project/tnl-lbm.git

3. Install the necessary tools and dependencies:
    - [CMake](https://cmake.org/) build system (version 3.24 or newer)
    - [CUDA](https://docs.nvidia.com/cuda/index.html) toolkit (version 11 or newer)
    - compatible host compiler (e.g. [GCC](https://gcc.gnu.org/) or
      [Clang](https://clang.llvm.org/))
    - [CUDA-aware][CUDA-aware MPI] MPI library – for distributed computing
      (tested with [OpenMPI](https://www.open-mpi.org/))
    - [ADIOS2](https://adios2.readthedocs.io/) – a unified high-performance I/O framework
    - [zlib](https://www.zlib.net/) (available in most Linux distributions)
    - [libpng](http://www.libpng.org/pub/png/libpng.html) (available in most Linux distributions)

   The following libraries will be fetched automatically by CMake if they are
   missing on your system:
    - [fmt](https://github.com/fmtlib/fmt/) – string formatting library
    - [spdlog](https://github.com/gabime/spdlog) – logging library
    - [nlohmann_json](https://github.com/nlohmann/json) – JSON parsing and dumping library
    - [argparse](https://github.com/p-ranav/argparse) – CLI argument parsing library

4. Configure the build using `cmake` in the root path of the Git repository:

       cmake -B build -S . <additional_configure_options...>

   This will use `build` in the current path as the build directory.
   The path for the `-S` option corresponds to the root path of the project.
   You may use additional options to configure the build:

   - `-DCMAKE_BUILD_TYPE=<type>` where `<type>` is one of `Debug`, `Release`,
     `RelWithDebInfo`
   - `-DCMAKE_CUDA_ARCHITECTURES=<arch>` – to build for a CUDA architecture
     other than "native"

5. Build the targets using `cmake`:

       cmake --build build

6. Run the example solver and supply its command-line arguments (here `4`
   determines the size of the lattice):

       ./build/sim_NSE/sim_1 4

   Distributed simulations can be run using `mpirun`. For example, to use two
   subdomains:

       mpirun -np 2 ./build/sim_NSE/sim_1 4

For convenience, steps 4-6 can be performed by running a simple script. For
example, to build and run `sim_1` as in the previous example:

    ./sim_NSE/run sim_1 4

## Getting involved

The TNL project welcomes and encourages participation by everyone. While most of the work for TNL
involves programming in principle, we value and encourage contributions even from people proficient
in other areas.

This section provides several areas where both new and experienced TNL users can contribute to the
project. Note that this is not an exhaustive list.

- Join the __code development__. Our [GitLab issues tracker][GitLab issues] collects ideas for
  new features, or you may bring your own.
- Help with __testing and reporting problems__. Testing is an integral part of agile software
  development which refines the code development. Constructive critique is always welcome.
- Contact us and __provide feedback__ on [GitLab][GitLab issues]. We are interested to know how
  and where you use TNL and the TNL-LBM module.

[GitLab issues]: https://gitlab.com/tnl-project/tnl-lbm/-/issues

## Citing

If you use TNL-LBM in your scientific projects, please cite the following papers in
your publications:

- J. Klinkovský, P. Eichler, R. Straka, T. Oberhuber, R. Fučík,
  [TNL-LBM: Scalable lattice Boltzmann method implementation based on Template Numerical Library](https://doi.org/10.1007/s11227-026-08292-0),
  Journal of Supercomputing 82, 167 (2026).
- R. Fučík, P. Eichler, R. Straka, P. Pauš, J. Klinkovský, and T. Oberhuber,
  [On optimal node spacing for immersed boundary–lattice Boltzmann method in 2D and 3D](https://doi.org/10.1016/j.camwa.2018.10.045),
  Computers & Mathematics with Applications 77.4 (2019), pages 1144–1162.
- R. Fučík, R. Galabov, P. Pauš, P. Eichler, J. Klinkovský, R. Straka, J. Tintěra, and R. Chabiniok,
  [Investigation of phase-contrast magnetic resonance imaging underestimation of turbulent flow through the aortic valve phantom: experimental and computational study using lattice Boltzmann method](https://doi.org/10.1007/s10334-020-00837-5),
  Magnetic Resonance Materials in Physics, Biology and Medicine 33.5 (2020), pages 649–662.
- P. Eichler, R. Fučík, and R. Straka,
  [Computational study of immersed boundary-lattice Boltzmann method for fluid-structure interaction](https://doi.org/10.3934/dcdss.2020349),
  Discrete & Continuous Dynamical Systems-S 14.3 (2021), page 819.
- P. Eichler, V. Fuka, and R. Fučík,
  [Cumulant lattice Boltzmann simulations of turbulent flow above rough surfaces](https://doi.org/10.1016/j.camwa.2021.03.016),
  Computers & Mathematics with Applications 92 (2021), pages 37–47.
- M. Beneš, P. Eichler, R. Fučík, J. Hrdlička, J. Klinkovský, M. Kolář, T. Smejkal, P. Skopec, J. Solovský, P. Strachota, R. Straka, and A. Žák,
  [Experimental and numerical investigation of air flow through the distributor plate in a laboratory-scale model of a bubbling fluidized bed boiler](https://doi.org/10.1007/s13160-022-00518-x),
  Japan Journal of Industrial and Applied Mathematics 39.3 (2022), pages 943–958.
- J. Klinkovský, A. C. Trautz, R. Fučík, T. H. Illangasekare,
  [Lattice Boltzmann method–based efficient GPU simulator for vapor transport in the boundary layer over a moist soil: Development and experimental validation](https://doi.org/10.1016/j.camwa.2023.02.021),
  Computers & Mathematics with Applications 138, 65-87 (2023).

## Authors

The code originates in the work of Robert Straka, Radek Fučík, and Pavel Eichler.
High-performance computational capabilities and interoperation with TNL were
developed by Jakub Klinkovský and Tomáš Oberhuber. The current code maintainer
is Jakub Klinkovský.

Furthermore, various features were developed in cooperation with students
working on their research projects at the [Faculty of Nuclear Sciences and
Physical Engineering](https://www.fjfi.cvut.cz/), [Czech Technical University
in Prague](https://www.cvut.cz/).

## License

TNL-LBM is provided under the terms of the [MIT License](./LICENSE).
