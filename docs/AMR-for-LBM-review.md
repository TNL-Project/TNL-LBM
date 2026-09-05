# Adaptive Mesh Refinement for the Lattice Boltzmann Method: A Systematic Review of Methods, Coupling Strategies, and Implementations

## Abstract

The Lattice Boltzmann Method (LBM) solves fluid dynamics by evolving particle distribution functions on a discrete velocity lattice, an approach valued for its algorithmic simplicity and suitability for massive parallelization. Its native formulation, however, ties spatial and temporal discretization together through the lattice velocity, restricting the method to uniform Cartesian grids. Adaptive Mesh Refinement (AMR) removes this restriction by locally refining and coarsening the grid in response to solution features, yielding substantial savings in memory and computation. This review surveys the evolution of AMR techniques for LBM from the foundational works of Filippova and Hänel (1998) and Chen (1998) through state-of-the-art GPU-native implementations. We identify five principal method families: cell-vertex (node-based), cell-centered (volumetric), combined, multiresolution (wavelet-based), and finite-difference. We compare their conservation guarantees, stability characteristics, and compatibility with collision operators. The analysis shows that cell-centered volumetric approaches provide exact mass and momentum conservation at grid interfaces and support arbitrary collision operators, while cell-vertex methods offer superior locality for parallel execution. Aeroacoustic applications remain the most demanding regime: all grid-coupling schemes emit spurious acoustic noise when vortical structures cross refinement interfaces, though the magnitude varies by up to an order of magnitude in power spectral density depending on the coupling strategy and collision model. On the implementation side, block-structured forest-of-octrees data structures have demonstrated scalability to trillions of cells on petascale systems, and GPU-native AMR, where mesh management resides entirely on the device, represents an emerging frontier. We conclude by identifying open problems: the absence of standardized benchmarks, multi-GPU dynamic load balancing, coupling of AMR with complex boundary geometries, and extension of multiresolution analysis to three-dimensional turbulent flows.

**Keywords:** lattice Boltzmann method, adaptive mesh refinement, grid refinement, multiresolution analysis, GPU acceleration, aeroacoustics, computational fluid dynamics

---

## 1. Introduction

The Lattice Boltzmann Method has become a widely used alternative to Navier-Stokes solvers for computational fluid dynamics (CFD). Rather than discretizing the macroscopic conservation equations, LBM evolves mesoscopic particle distribution functions on a regular lattice via a "collide-and-stream" algorithm. Each time step consists of a local collision relaxation followed by an explicit propagation of distributions to neighboring lattice sites. This structure (local, explicit, and free of pressure-Poisson solves) makes LBM inherently parallelizable, a property that has been exploited in simulations reaching trillions of grid cells on petascale supercomputers (Schornbaum & Rüde, 2018).

The method's algorithmic elegance comes at a cost. The standard LBM couples the spatial step $\Delta x$ and the temporal step $\Delta t$ through the lattice velocity $c = \Delta x / \Delta t$, which is fixed by the discrete velocity set (e.g., D2Q9, D3Q19, D3Q27). This coupling restricts the method to uniform Cartesian grids with equidistant spacing. When a simulation requires high resolution near a wall, around an obstacle, or in a region of strong shear, the uniform-grid requirement forces the same fine spacing across the entire domain. The result is wasted computation in regions where coarse resolution would suffice.

Adaptive Mesh Refinement addresses this inefficiency by refining the grid locally, only where the solution demands it, and coarsening elsewhere. First proposed for LBM by Filippova and Hänel (1998) and independently by Chen (1998), AMR for LBM has matured over three decades into a field with distinct method families, each with its own conservation properties, stability characteristics, and implementation trade-offs. The challenge is particularly acute for LBM: unlike finite-volume AMR, where conservation is enforced through consistent fluxes at cell faces, LBM's distribution-function formulation requires interpolation and rescaling at every lattice velocity direction across grid interfaces, making conservation guarantees non-trivial (Guzik et al., 2014).

A recent comprehensive review by An, Chen, and Bergadà (2025) classifies all grid technologies for LBM into six categories: body-fitted, non-uniform rectangular, multigrid, hierarchical Cartesian (octree/quadtree), unstructured, and meshless. The hierarchical Cartesian category, which preserves LBM's local uniformity at each grid level while allowing global non-uniformity, dominates the AMR-for-LBM literature and is the focus of this review.

This paper systematically surveys AMR techniques for LBM to answer three questions:

- **RQ1:** What are the main algorithmic categories of AMR for LBM?
- **RQ2:** How do these methods handle grid-interface coupling to conserve mass, momentum, and stability?
- **RQ3:** What accuracy and performance gains have been demonstrated relative to uniform-grid LBM?

Section 2 provides background on LBM and AMR. Section 3 presents the method taxonomy. Section 4 analyzes grid-interface coupling in detail. Section 5 surveys parallel and GPU implementations. Section 6 provides a comparative analysis. Section 7 discusses unresolved challenges, and Section 8 concludes.

## 2. Background

### 2.1 The Lattice Boltzmann Method

LBM discretizes the Boltzmann equation in velocity space using a finite set of discrete velocities $\{c_i\}_{i=0}^{q-1}$ (Krüger et al., 2017). The evolution equation for the distribution function $f_i$ at position $x$ and time $t$ is:

$$f_i(x + c_i \Delta t, t + \Delta t) - f_i(x, t) = \Omega_i(f)$$

where $\Omega_i$ is the collision operator. The most common collision models include the single-relaxation-time BGK operator, multiple-relaxation-time (MRT) operators, and regularized operators such as the Hybrid Recursive Regularized (HRR) model (Astoul et al., 2021a). Macroscopic quantities — density $\rho$, velocity $u$, and viscous stress $\sigma$ — are obtained as moments of the distribution functions.

The collision step is purely local: each cell updates its distributions based on its own state. The streaming step propagates distributions to neighboring cells along the lattice directions. This locality is the source of LBM's parallel scalability. It also means that the method is naturally suited to Cartesian grids, where the streaming operation is a simple, regular memory access pattern.

### 2.2 Adaptive Mesh Refinement

AMR, as introduced by Berger and Oliger (1984) and later formalized by Berger and Colella (1989) for finite-volume methods, adaptively refines the computational grid in regions requiring higher resolution and coarsens it elsewhere. The approach maintains a hierarchy of grid levels, each with uniform spacing, related by a refinement ratio (typically 2). At interfaces between levels, special coupling procedures transfer information between grids of different resolution.

For LBM, the refinement ratio of 2 has a natural justification: the lattice velocity $c$ is fixed, so halving $\Delta x$ halves $\Delta t$ as well. One coarse time step therefore corresponds to two fine time steps. This 2:1 ratio is enforced as a "balance" constraint in most implementations: neighboring blocks may differ by at most one refinement level.

### 2.3 Why AMR for LBM Is Hard

Three properties of LBM make AMR more challenging than for finite-volume methods:

1. **Distribution-function coupling**: LBM evolves distribution functions, not macroscopic fluxes. At a grid interface, each lattice velocity direction requires its own interpolation and rescaling. There is no single "flux at the face" to make consistent.

2. **Non-equilibrium rescaling**: The non-equilibrium part of the distribution function depends on the grid spacing through the relaxation time $\tau$. Transferring distributions between grid levels requires rescaling the non-equilibrium component to maintain the correct viscosity (Filippova & Hänel, 1998).

3. **Non-hydrodynamic modes**: LBM supports non-physical (non-hydrodynamic) modes that propagate at speeds different from the fluid velocity. When these modes cross a refinement interface, they generate spurious vorticity and acoustic noise (Astoul et al., 2021a). This problem is intrinsic to resolution changes and cannot be fully eliminated by any grid-coupling algorithm alone.

## 3. Method Taxonomy

The literature can be organized along a clear taxonomy of AMR approaches for LBM, spanning three axes: grid layout, adaptivity paradigm, and implementation strategy.

### 3.1 Cell-Vertex (Node-Based) Methods

Cell-vertex methods, also called node-based approaches, place grid nodes at cell corners such that coarse and fine grid nodes partially co-locate along refinement interfaces. The foundational algorithm was developed by Filippova and Hänel (1998), who proposed rescaling the non-equilibrium part of the distribution function to ensure continuity of density, momentum, and viscous stress across grid levels. Their method preserves the second-order accuracy of the original LBM and was validated on flow past a cylinder at low to moderate Reynolds numbers.

Filippova and Hänel (2000) extended this work to enable variable time-step ratios between grid levels, reducing the number of time steps required on refined grids without compromising accuracy. The gain in computational time was found to be significant.

The cell-vertex approach has a natural advantage: at co-located nodes, information can be transferred directly without spatial interpolation. For non-co-located fine nodes, spatial interpolation from nearby coarse nodes is required.

A critical improvement came from Lagrava, Malaspinas, Latt, and Chopard (2012), who introduced spatial filtering of fine-grid distribution functions before transfer to the coarse grid. Fine grids resolve scales that coarse grids cannot represent. Without filtering, these unresolved scales cause aliasing when projected onto the coarse grid, leading to numerical instability in turbulent flows. Lagrava (2012) provided a detailed analysis in his doctoral thesis, validated on both laminar and turbulent problems in two and three dimensions using the open-source Palabos library (Latt et al., 2021).

The principal limitation of cell-vertex methods is conservation. Rescaling the non-equilibrium distribution locally preserves mass and momentum at co-located nodes but does not guarantee global conservation across the interface (Guzik et al., 2014). Additionally, the original Filippova-Hänel rescaling is tied to the BGK collision operator and does not directly generalize to MRT or regularized models.

### 3.2 Cell-Centered (Volumetric) Methods

Cell-centered approaches place grid nodes at cell centers, eliminating co-location between coarse and fine grid points. The foundational volumetric formulation was introduced by Chen (1998), who reformulated LBM in terms of particle masses moving between cells of different resolution. By treating distributions as masses rather than densities, conservation of mass and momentum at grid interfaces is guaranteed by construction.

Chen et al. (2006) formalized the volumetric grid refinement concept, showing that conservation laws are exactly guaranteed through the volumetric formulation. Their approach, with a refinement factor of 2, has been used in the commercial PowerFLOW solver for nearly two decades.

Guzik et al. (2014) developed an alternative cell-centered approach within the Chombo AMR framework (Adams et al., 2015), based on the Colella-Berger methodology for finite-volume CFD. They proposed a space-time interpolation to fill ghost cells on the fine grid, solving constrained least-squares problems to ensure mass conservation. Their approach offers two coupling strategies:

- **Initial-Value Problem (IVP)**: Sufficient ghost cells are filled at the base time $t_\ell$ so that the domain of dependence is fully populated for all fine-grid subcycles. This requires $n_{\text{ref}}$ layers of ghost cells.
- **Boundary-Value Problem (BVP)**: Only one ghost-cell layer is required, but it must be filled with new information before each fine subcycle. This approach avoids perturbing steady-state solutions.

The BVP approach achieves smaller solution errors than purely spatial interpolation methods, as demonstrated through Taylor-Green vortex and Karman vortex street benchmarks using the D3Q19 lattice.

Freitas, Meinke, and Schröder (2006) developed a cell-centered refinement approach where the transformation and interpolation operations are formulated independently of the applied LBM scheme. Their nonlinear interpolation calculates missing fine-grid distributions in two steps: bilinear (2D) or trilinear (3D) interpolation to a virtual coarse cell, followed by transformation from coarse to fine grid. This work established a research lineage at RWTH Aachen.

Yu and Fan (2009) extended multi-block AMR to two-phase flow using the Shan-Chen interaction potential model, introducing "explode" and "coalesce" operations at refinement jumps for coarse-to-fine and fine-to-coarse transfers. Fakhari, Geier, and Lee (2016) further advanced two-phase AMR with a mass-conserving block-structured method using biquadratic interpolation for buffer-layer filling and a phase-field gradient as the refinement criterion, reporting 18–23× speedup and 20× memory savings compared to uniform grids.

### 3.3 Combined Methods

A third category uses a mixed arrangement: fine interface nodes reside in cell centers while coarse nodes sit in cell corners (Schukmann et al., 2023). This layout has an overlapping width of two coarse cells and employs gradient-based compact interpolation to preserve second-order accuracy. The combined approach was developed to merge the conservation properties of cell-centered methods with the locality advantages of cell-vertex approaches. Geier et al. (2009) introduced the concept of moment-based compact interpolation (bubble functions) exploiting second-order central moments stored in the distribution function, and Schönherr et al. (2011) developed it into a compact interpolation method using statistical moments for non-uniform grids on CPUs and GPUs. Qi et al. (2019) extended the compact interpolation to three dimensions and implemented it in the Musubi solver (Hasert et al., 2014), demonstrating that only four source elements from the adjacent level are needed in both 2D and 3D, improving computational efficiency.

### 3.4 Multiresolution (Wavelet-Based) Methods

Bellotti, Gouarin, Graille, and Massot introduced a fundamentally different paradigm connecting LBM to adaptive multiresolution (MR) analysis based on wavelet theory (Bellotti et al., 2022a, 2022b). Unlike heuristic AMR, where mesh refinement is guided by user-defined criteria such as vorticity magnitude or gradient sensors, MR decomposes the solution onto a local wavelet basis whose coefficients provide a precise measure of local regularity.

The MR approach provides rigorous error control, problem-independence, and preservation of the original LBM scheme:

1. **Rigorous error control**: The additional error introduced by mesh adaptation is bounded by a user-specified tolerance through the Harten heuristics (Bellotti et al., 2022a).
2. **Problem-independence**: Refinement is driven by solution regularity alone, not problem-specific sensors.
3. **Preservation of the original LBM scheme**: The collision phase is unaffected, and the method works for any LBM scheme without modification.

Bellotti et al. (2022a) established the method for one-dimensional hyperbolic conservation laws with a formal error analysis. Bellotti et al. (2022b) extended it to multidimensional problems for both parabolic and hyperbolic systems, demonstrating significant memory compression for solutions with localized structures. Bellotti et al. (2023) performed an equivalent-equations analysis showing that MR-based mesh adaptation preserves the target equations at high accuracy, with collision strategy having only marginal impact on solution quality.

The MR approach's principal limitation is implementation complexity. It requires specialized data structures (the SAMURAI open-source platform; Bellotti et al., 2022b) and has been tested primarily on relatively simple problems. Its applicability to high-Reynolds-number turbulent flows and complex geometries remains unproven.

### 3.5 Finite-Difference AMR

Fakhari and Lee (2014) presented an AMR algorithm for the finite-difference lattice Boltzmann method (FDLBM). Their approach removes the need for a tree-type data structure by using pointer attributes to determine block neighbors. Because the streaming process is formulated in Eulerian (finite-difference) form, no rescaling of distribution functions or temporal interpolation is needed at fine-coarse grid boundaries. The method was validated on Taylor-Green vortex flow, lid-driven cavity flow, thin shear layer flow, and flow past a square cylinder, all vorticity-dominated flows.

## 4. Grid-Interface Coupling

The grid interface — where coarse and fine grids meet — is the heart of any AMR-for-LBM algorithm. The quality of the coupling procedure determines conservation, stability, accuracy, and acoustic properties of the overall scheme.

### 4.1 Conservation Guarantees

A central distinction between method families is their conservation guarantee. As discussed in Sections 3.1 and 3.2, cell-vertex methods ensure local continuity of macroscopic quantities ($\rho$, $\rho u$, $\sigma_{\alpha\beta}$) at co-located nodes but do not guarantee global mass conservation, while cell-centered volumetric methods guarantee global mass and momentum conservation through particle redistribution. The combined method achieves local continuity through gradient-based compact interpolation, though global mass conservation is not guaranteed (Qi et al., 2019 report that the compact interpolation algorithm violates mass conservation in domains without Dirichlet boundary conditions).

The conservation challenge is specific to LBM's distribution-function formulation: unlike finite-volume methods, where conservation is enforced by consistent face fluxes, LBM requires conservation at each lattice velocity direction near the interface (Guzik et al., 2014). As noted in Section 3.2, the volumetric approach's independence from the collision step allows the method to be used with any LBM scheme, in contrast to cell-vertex methods where the Filippova-Hänel rescaling is tied to the BGK operator.

### 4.2 Stability and the Role of Filtering

Grid refinement interfaces introduce stability challenges beyond those of uniform-grid LBM. Schukmann et al. (2023) conducted systematic stability comparisons of cell-vertex, cell-centered, and combined approaches using four collision models (BGK, MRT, RR, HRR) for square duct flow. Their key findings:

- Cell-vertex approaches qualitatively emit less spurious acoustic noise than cell-centered layouts.
- The HRR collision model, originally designed for cell-vertex grids, requires adaptation for cell-centered and combined layouts to ensure consistent use of central finite differences.
- Stability limits depend on the specific coupling mechanism, not just the grid layout.

As noted in Section 3.1, Lagrava et al. (2012) introduced spatial filtering of fine-grid distributions before transfer to the coarse grid. This filtering is mandatory at fine-to-coarse transfer locations: without it, unresolved scales cause aliasing and instability. This is especially critical for aeroacoustic applications, where added artificial bulk viscosity would corrupt the acoustic prediction.

### 4.3 Interpolation Order

Multiple studies confirm that the interpolation order at grid interfaces is critical for maintaining LBM's second-order accuracy. The Palabos implementation (Lagrava et al., 2012) uses linear (second-order) temporal interpolation and third- or fourth-order spatial interpolation. Gendre et al. (2017) confirmed that at least third-order spatial and temporal interpolation is required to maintain consistency with the discrete velocity Boltzmann equation in their directional splitting approach.

The moment-based compact interpolation method (Geier et al., 2009; Schönherr et al., 2011; Schönherr, 2015) provides second-order accurate coupling at grid interfaces. Qi, Klimach, and Roller (2019) extended the method to three dimensions within the Musubi solver, demonstrating second-order convergence for both velocity and strain rate in the Taylor-Green vortex test case with only four source elements from the adjacent level in both 2D and 3D. An earlier precursor to this approach was the bubble function interpolation of Geier, Greiner, and Korvink (2009), which exploited second-order central moments ($\kappa_{xy}$, $\kappa_{xx}-\kappa_{yy}$) stored in the distribution function to achieve quadratic interpolation from only four neighboring nodes, validated on vortex shedding behind a plate with the cascaded collision operator.

Kutscher, Geier, and Krafczyk (2018) extended compact quadratic interpolation to the cumulant LBM (Geier et al., 2015) on staggered grids, using 33 coefficients determined from nodal velocities and velocity gradients computed from pre-collision cumulants. They demonstrated that quadratic interpolation is necessary at grid interfaces: linear interpolation introduces spurious numerical viscosity, as shown via a channel flow benchmark. Their method was applied to turbulent flow over porous media with up to six refinement levels and approximately 700 million grid points.

As discussed in Section 3.2, Freitas et al. (2006) developed a scheme-independent nonlinear interpolation approach for cell-centered refinement, establishing a research lineage later extended by Schukmann et al. (2023, 2025).

### 4.4 Aeroacoustic Challenges

Aeroacoustic applications are the most demanding regime for AMR-for-LBM. Acoustic pressure fluctuations are several orders of magnitude smaller than hydrodynamic ones. Even tiny errors at grid interfaces generate spurious acoustic waves that can contaminate the entire computational domain.

Gendre et al. (2017) were the first to systematically address aeroacoustic grid refinement for LBM. Their directional splitting approach couples finite-difference and lattice Boltzmann methods locally at the interface, better accounting for gradients normal to the interface than standard cell-vertex algorithms. Their method reduces spurious acoustic noise power by a factor of approximately 55 (measured via power spectral density of density fluctuations near the refinement interface) for convective Mach numbers from 0.04 to 0.2, compared to the Lagrava et al. (2012) algorithm. They also showed that an increased free-stream Mach number (within the weakly compressible regime) strongly deteriorates the spurious noise situation, although the magnitude may remain negligible for purely aerodynamic studies.

Astoul et al. (2021a) identified the root cause of spurious noise: non-hydrodynamic modes inherent to LBM generate spurious vorticity and acoustics when projected onto coarser grids. This phenomenon is intrinsic to resolution changes (aliasing) and is independent of the specific grid-coupling algorithm. Their solution: choose an appropriate collision model — the Hybrid Recursive Regularized (HRR) model — to filter non-hydrodynamic mode contributions regardless of the grid-coupling algorithm. They validated this approach on a convected vortex and a turbulent flow around a cylinder, obtaining large reductions in both spurious noise and vorticity.

Astoul et al. (2021b) proposed a direct-coupling cell-vertex algorithm that eliminates the overlapping mesh layer used in conventional methods. By solving a non-linear equation system constraining zeroth- and first-order non-equilibrium moments, the method establishes a tighter link between fine and coarse grids. The direct coupling reduces duplicated points at grid interfaces (beneficial for parallelization) and improves aeroacoustic accuracy. The method was validated on an acoustic pulse, a convected vortex, and a turbulent circular cylinder wake flow at high Reynolds number.

Feng et al. (2020) also extended grid refinement to three-dimensional compressible aerodynamics within the HRR-LBM framework, demonstrating isentropic vortex propagation through transition interfaces, shock-vortex interaction with intersection between grid refinement interface and shock corrugation, and transonic flow over a three-dimensional DLR-M6 wing with seven levels of grid refinement.

Schukmann et al. (2025) conducted the first systematic comparison of spurious aeroacoustic emissions across cell-vertex, cell-centered, and combined layouts. Their findings, validated on four benchmark cases (2D Gaussian acoustic pulse, 1D convected acoustic wave, 2D convected barotropic vortex, and 3D jet flow), show that cell-centered approaches (with either linear or uniform explosion during the coarse-to-fine coupling) and vertex-centered direct-coupling methods produce the least spurious noise.

For thermal and compressible flows, Frapolli (2017) extended node-centered grid refinement to entropic LBM in his doctoral thesis, deriving rescaling procedures for the equilibrium and non-equilibrium populations that maintain continuity of macroscopic quantities across grid interfaces. Using the entropic stabilizer, shock waves were shown to cross refinement interfaces without spurious reflections, as demonstrated on supersonic NACA0012 airfoil simulations at Mach 1.5.

## 5. Parallel and GPU Implementations

### 5.1 Block-Structured AMR on CPU Clusters

Schornbaum and Rüde (2016, 2018) developed the most scalable AMR-for-LBM implementation to date within the waLBerla framework (Bauer et al., 2021). Their approach uses a 2:1-balanced forest-of-octrees domain partitioning where octrees are implicitly defined via globally unique block identifiers — no explicit parent-child storage is needed. Every block is aware of all its spatially adjacent neighbors, creating a distributed adjacency graph.

Key innovations include:

- **Fully distributed data structures**: Per-process memory for metadata is bounded independently of processor count, enabling unlimited scalability.
- **Lightweight proxy for load balancing**: A temporary shallow copy of the core data structure, containing only topological information, enables inexpensive, local, diffusion-based dynamic load balancing without global synchronization.
- **Volume-based refinement**: Combining volumetric grid refinement with Chen et al.'s (2006) interpolation scheme.

On an IBM Blue Gene/Q system, they demonstrated weak scalability with up to 1.8 million threads and close to one trillion grid cells, achieving near-perfect scalability. Strong scaling reached 1,000 time steps per second for 8.5 million cells (Schornbaum & Rüde, 2016). The dynamic repartitioning overhead was measured at a small fraction of total simulation runtime, with an entire AMR cycle completing in approximately 3.5 seconds on 458,752 cores for 197 billion cells (Schornbaum & Rüde, 2018).

### 5.2 GPU-Accelerated AMR

Schönherr et al. (2011) presented an early multi-thread LBM implementation on non-uniform grids for both CPUs and GPUs, using second-order compact interpolation at grid interfaces with a grid spacing ratio of 2 and acoustic time scaling. Their hierarchical data structure (node → block → patch → domain) supported arbitrary recursive nesting of refinement levels, and the interface interpolation introduced almost no overhead compared to uniform-grid LBM, as validated on flow past a cylinder.

Hsu, Chang, and Smith (2018) implemented multi-block AMR for LBM on GPUs. Their approach divides the domain into uniform mesh blocks, each processed with conventional GPU-LBM with minimal modification. The multi-block scheme yielded similar results to fine uniform-grid simulations while requiring only 25 to 30 percent of the time. For 3D flow over a square cylinder, a maximum speedup ratio of 55 was achieved compared to a serial CPU code.

Onodera et al. (2018a) developed CityLBM for real-time wind simulation on GPU supercomputers (TSUBAME3.0). Using block-structured AMR with a forest-of-octrees approach, they reduced total grid points to less than 10 percent compared to a fine uniform grid. A single NVIDIA Tesla P100 GPU achieved 383.3 MLUPS (million lattice updates per second) — 16 times faster than a CPU process. Their Communication-Reduced Multi-Time-step (CRMT) algorithm (Onodera et al., 2018b) reduced communication costs by approximately 64 percent, enabling real-time simulation of a approximately 2 km square area at 1 m resolution using approximately 200 GPUs.

Schönherr (2015) developed a grid refinement method with moment-based compact interpolation for multi-GPGPU LBM simulations in his doctoral thesis, demonstrating up to 5 refinement levels with 23.6 million nodes for a ship-propeller flow.

### 5.3 GPU-Native AMR

A significant limitation of GPU-accelerated AMR is that mesh management and adaptation typically occur on the CPU, necessitating frequent CPU-GPU data transfers. Jaber, Essel, and Sullivan (2025a) developed a GPU-native AMR framework (AGAL) where the entire mesh structure — including refinement, coarsening, and 2:1 balancing — is managed on the GPU. Their block-based forest-of-octrees approach uses integer index arrays to identify octree nodes, enabling data-parallel mesh adaptation via Thrust, a C++ parallel algorithms library for CUDA.

The implementation includes an LBM solver for weakly compressible flow, validated on lid-driven cavity and flow past a square cylinder benchmarks across multiple velocity sets (D2Q9, D3Q19, D3Q27) in single and double precision. Linear and cubic interpolation communicate data from coarse to fine grids along refinement interfaces, while basic averaging transfers data in the reverse direction. Tests on consumer and datacenter-grade GPUs demonstrated versatility across hardware platforms.

Jaber et al. (2025b) extended this framework with GPU-native embedding of complex geometries (triangle meshes) through solid voxelization with local ray casting, accelerated by a hierarchy of spatial bins. A flattened lookup table of cut-link distances supports accurate interpolated bounce-back boundary conditions. The method was validated on external flows past a circular/square cylinder (2D, Re = 100) and a sphere (3D, Re = 10, 15, 20).

The MARBLES solver (Multi-scale Adaptively Refined Boltzmann LatticE Solver), developed at the National Renewable Energy Laboratory (NREL) and built on AMReX (Zhang et al., 2020), provides an alternative approach supporting multiple GPU architectures (Intel, AMD, NVIDIA) through performance portability. It targets multi-scale flows in complex media using block-structured AMR with various parallel decomposition strategies including MPI and OpenMP.

## 6. Comparative Analysis

Table 1 summarizes the key properties of each method family.

**Table 1.** Comparison of AMR method families for LBM.

| Property | Cell-Vertex | Cell-Centered | Combined | Multiresolution | FDLBM |
|---|---|---|---|---|---|
| Global mass conservation | Not guaranteed | Guaranteed | Not guaranteed | Guaranteed (wavelet decomposition) | Not guaranteed |
| Collision operator compatibility | BGK (original); MRT/HRR in later works | Any | Any | Any | Any |
| Spatial interpolation at interface | 3rd–4th order | Linear, uniform, or space-time (Guzik et al.) | Compact gradient | Wavelet prediction | Not needed (Eulerian formulation) |
| Temporal interpolation | Required | IVP: pre-filled; BVP: per-subcycle | Not required | Not required | Not needed |
| Error control | Heuristic | Heuristic | Heuristic | Rigorous (wavelet bounds) | Heuristic |
| Locality for parallelism | High | Medium | Medium | Low (complex data structure) | High |
| Aeroacoustic suitability | Moderate (DC variant best) | Good (least noise) | Moderate | Untested | Untested |
| GPU-native implementation | Not reported | Yes (AGAL, MARBLES) | Not reported | Not reported | Not reported |
| Maximum demonstrated scale | Not demonstrated at petascale | ~10^12 cells (waLBerla) | — | Single-node (SAMURAI) | Single-block (2D/3D benchmarks) |

Note: DC = direct coupling (Astoul et al., 2021b).

Reported performance gains vary widely by application:

- **Memory reduction**: Hierarchical Cartesian grids reduce total cells by approximately 95 percent compared to uniform grids (An et al., 2025). Forest-of-octrees AMR reduces to under 10 percent of fine uniform grid point counts (Onodera et al., 2018a).
- **Speedup**: Multi-block GPU AMR achieves 25 to 30 percent of uniform fine-grid runtime (Hsu et al., 2018). GPU-native AMR achieves order-of-magnitude speedup over CPU implementations (Jaber et al., 2025a).
- **Scalability**: waLBerla demonstrates near-perfect weak scaling to 1.8 million threads (Schornbaum & Rüde, 2018). CityLBM achieves 93 percent parallel efficiency on 196 GPUs (Onodera et al., 2018b).

## 7. Discussion

### 7.1 Convergence of Method Families

The three grid-layout families are converging. Cell-vertex methods have adopted filtering of fine-grid distributions to prevent aliasing at coarse-grid interfaces (Lagrava et al., 2012). Cell-centered methods have incorporated moment-based compact interpolation from combined approaches (Geier et al., 2009; Schönherr et al., 2011; Qi et al., 2019). The HRR collision model, originally specific to cell-vertex grids, has been adapted to all three layouts (Schukmann et al., 2023). The practical distinction is increasingly about implementation strategy — data structures, parallelization, and GPU residency — rather than fundamental algorithmic differences.

### 7.2 Unresolved Challenges

Several gaps persist in the literature:

**No standardized benchmarks.** Studies use different test cases (lid-driven cavity, cylinder flow, Taylor-Green vortex, duct flow), making cross-study comparison difficult. Only Schukmann et al. (2023, 2025) provide systematic head-to-head comparisons of multiple methods on identical test cases. The choice of refinement criterion itself is also non-standardized: most studies use heuristic sensors such as vorticity magnitude or gradient thresholds, while Thorimbert et al. (2022) proposed a kinetic-based sensor exploiting the ratio of off-equilibrium to equilibrium distribution functions (scaling with the local Knudsen number), validated in both Palabos (vertex-centered static refinement) and AMROC-LBM (cell-centered dynamic AMR) across incompressible, compressible, and multiphase regimes. A community-accepted benchmark suite and refinement criteria would accelerate progress.

**Aeroacoustic coupling.** While significant progress has been made (Gendre et al., 2017; Astoul et al., 2021a, 2021b), no method fully eliminates spurious noise at refinement interfaces. The fundamental cause — non-hydrodynamic mode aliasing — is intrinsic to resolution changes. Current best practice combines an appropriate collision model (HRR) with a low-noise coupling strategy (cell-centered or direct-coupling), but residual spurious emissions remain.

**Dynamic AMR on GPUs.** GPU-native AMR (Jaber et al., 2025a) is recent and limited to a single GPU. Extending to multi-GPU with dynamic load balancing remains an open problem. The MARBLES solver supports multi-GPU but uses CPU-managed mesh adaptation. The trade-off between GPU-resident mesh management (eliminating transfers) and CPU-managed mesh management (simpler, mature load balancing) is not yet resolved.

**Complex geometries.** Most AMR-for-LBM work assumes axis-aligned boundaries. Jaber et al. (2025b) represents a first step toward complex geometry embedding in GPU-native AMR, but the approach has been validated only on relatively simple external flow geometries. Coupling AMR with immersed boundary methods for moving boundaries is largely unexplored.

**Multiresolution maturity.** The MR approach (Bellotti et al., 2022a, 2022b) offers the most rigorous error control but has been tested primarily on scalar and system conservation laws in one and two dimensions. Its applicability to three-dimensional turbulent flows and complex geometries is unproven. The SAMURAI implementation platform, while providing efficient data structures, has not been parallelized to the scale of waLBerla or AMReX.

### 7.3 Limitations of This Review

- **Heterogeneous test cases**: Performance and accuracy metrics are not directly comparable across studies due to different flow problems, grid configurations, and hardware.
- **Publication bias**: Studies report positive speedup results. Cases where AMR overhead exceeded gains are likely underreported.
- **Language bias**: Only English-language sources were included. Significant LBM research exists in Chinese and Japanese.
- **Access limitations**: Some paywalled articles could not be fully accessed, limiting detailed analysis of their methods.

## 8. Conclusion

AMR for LBM has evolved from the foundational cell-vertex rescaling of Filippova and Hänel (1998) into a diverse ecosystem of methods with distinct trade-offs. For applications prioritizing conservation and collision-operator flexibility, cell-centered volumetric methods (Chen, 1998; Guzik et al., 2014) are the recommended choice. For aeroacoustic applications, the direct-coupling cell-vertex approach with HRR collision (Astoul et al., 2021b) or cell-centered methods with linear explosion (Schukmann et al., 2025) minimize spurious noise. For extreme-scale parallel simulations, the waLBerla framework (Schornbaum & Rüde, 2018) provides proven petascale scalability. For rigorous error control on problems with localized features, the multiresolution approach (Bellotti et al., 2022b) is the only method with formal error bounds.

Future research should focus on four areas: (1) standardized benchmarks for AMR-for-LBM method comparison, (2) multi-GPU-native dynamic AMR with load balancing, (3) coupling of AMR with immersed boundary methods for complex and moving geometries, and (4) extension of multiresolution analysis to three-dimensional turbulent flows.

---

## Declarations

### Ethics Declaration

This is a computational methods review paper. No human or animal subjects were involved. No sensitive or dual-use data are presented.

### Conflict of Interest

The author declares no competing interests.

### Funding

This research received no external funding.

### Author Contributions (CRediT)

**Conceptualization:** Method design and research question formulation. **Investigation:** Literature search and source verification. **Writing — original draft:** AI-assisted draft preparation with author review and revision. **Writing — review and editing:** Revision and quality checks. All other contributions by the author.

### Data Availability

No new data were generated. All cited sources are publicly available through their respective DOIs or repositories.

### AI Disclosure Statement

This literature review was compiled with the assistance of AI-assisted research tools (OhMyOpenCode deep-research skill). The AI tool facilitated literature search, source verification, synthesis, and manuscript drafting. All cited sources were independently verified through web searches. The author reviewed all content for accuracy and takes full responsibility for the final manuscript.

---

## References

Adams, M., Colella, P., Graves, D. T., Johnson, J. N., Keen, N. D., Kirkby, M. J., & Sternberg, K. S. (2015). Chombo software package for AMR applications. *Lawrence Berkeley National Laboratory*. (Technical report)

An, B., Chen, K., & Bergadà, J. M. (2025). Grid technologies in lattice Boltzmann method: A comprehensive review. *Mathematics, 13*(17), 2861. https://doi.org/10.3390/math13172861

Feng, Y., Guo, S., Jacob, J., & Sagaut, P. (2020). Grid refinement in the three-dimensional hybrid recursive regularized lattice Boltzmann method for compressible aerodynamics. *Physical Review E, 101*, 063302. https://doi.org/10.1103/PhysRevE.101.063302

Astoul, T., Wissocq, G., Sengissen, A., Boussuge, J.-F., & Sagaut, P. (2021a). Analysis and reduction of spurious noise generated at grid refinement interfaces with the lattice Boltzmann method. arXiv preprint arXiv:2004.11863 [physics.comp-ph].

Astoul, T., Wissocq, G., Sengissen, A., Boussuge, J.-F., & Sagaut, P. (2021b). Lattice Boltzmann method for computational aeroacoustics on non-uniform meshes: A direct grid coupling approach. *Journal of Computational Physics, 430*, 110667. https://doi.org/10.1016/j.jcp.2021.110667

Bauer, M., Eibl, S., Godenschwager, C., Kohl, N., Kuron, M., Rettinger, C., Schornbaum, F., Schwarzmeier, C., Thönnes, D., Köstler, H., & Rüde, U. (2020). waLBerla: A block-structured high-performance framework for multiphysics simulations. *Computers & Mathematics with Applications, 81*, 423–451. https://doi.org/10.1016/j.camwa.2020.01.007

Bellotti, T., Gouarin, L., Graille, B., & Massot, M. (2022a). Multiresolution-based mesh adaptation and error control for lattice Boltzmann methods with applications to hyperbolic conservation laws. *SIAM Journal on Scientific Computing, 44*(4), C223–C259. https://doi.org/10.1137/21M140256X

Bellotti, T., Gouarin, L., Graille, B., & Massot, M. (2022b). Multidimensional fully adaptive lattice Boltzmann methods with error control based on multiresolution analysis. *Journal of Computational Physics, 471*, 111670. https://doi.org/10.1016/j.jcp.2022.111670

Bellotti, T., Gouarin, L., Graille, B., & Massot, M. (2023). High accuracy analysis of adaptive multiresolution-based lattice Boltzmann schemes via the equivalent equations. *SMAI Journal of Computational Mathematics, 9*, 119–156. https://doi.org/10.5802/smai-jcm.83

Berger, M. J., & Colella, P. (1989). Local adaptive mesh refinement for shock hydrodynamics. *Journal of Computational Physics, 82*(1), 64–84. https://doi.org/10.1016/0021-9991(89)90035-1

Berger, M. J., & Oliger, J. (1984). Adaptive mesh refinement for hyperbolic partial differential equations. *Journal of Computational Physics, 53*(3), 484–512. https://doi.org/10.1016/0021-9991(84)90073-1

Chen, H. (1998). Volumetric formulation of the lattice Boltzmann method for fluid dynamics: Basic concept. *Physical Review E, 58*(3), 3955–3963. https://doi.org/10.1103/PhysRevE.58.3955

Chen, H., Filippova, O., Hoch, J., Molvig, K., Shock, R., Teixeira, C., & Zhang, R. (2006). Grid refinement in lattice Boltzmann methods based on volumetric formulation. *Physica A: Statistical Mechanics and its Applications, 362*(1), 158–167. https://doi.org/10.1016/j.physa.2005.09.036



Schukmann, A., Schneider, A., Haas, V., & Böhle, M. (2023). Analysis of hierarchical grid refinement techniques for the lattice Boltzmann method by numerical experiments. *Fluids, 8*(3), 103. https://doi.org/10.3390/fluids8030103

Schukmann, A., Haas, V., & Schneider, A. (2025). Spurious aeroacoustic emissions in lattice Boltzmann simulations on non-uniform grids. *Fluids, 10*(2), 31. https://doi.org/10.3390/fluids10020031



Fakhari, A., Geier, M., & Lee, T. (2016). A mass-conserving lattice Boltzmann method with dynamic grid refinement for immiscible two-phase flows. *Journal of Computational Physics, 315*, 434–457. https://doi.org/10.1016/j.jcp.2016.03.058

 Grid refinement for lattice-BGK models. *Journal of Computational Physics, 147*(1), 219–228. https://doi.org/10.1006/jcph.1998.6089

Filippova, O., & Hänel, D. (2000). Acceleration of lattice-BGK schemes with grid refinement. *Journal of Computational Physics, 165*(2), 407–429. https://doi.org/10.1006/jcph.2000.6617

Frapolli, N. (2017). *Entropic lattice Boltzmann models for thermal and compressible flows* (Doctoral thesis, ETH Zurich). https://doi.org/10.3929/ethz-a-010890892

 Turbulence simulation via the lattice-Boltzmann method on hierarchically refined meshes. In P. Wesseling, E. Oñate, & J. Périaux (Eds.), *Proceedings of the European Conference on Computational Fluid Dynamics (ECCOMAS CFD 2006)*, Egmond aan Zee, The Netherlands, September 5–8, 2006. TU Delft Repository. https://repository.tudelft.nl/record/uuid:94ae681d-f2a3-456d-b95d-a531210d3a7d

Geier, M., Greiner, A., & Korvink, J. G. (2009). Bubble functions for the lattice Boltzmann method and their application to grid refinement. *European Physical Journal Special Topics, 171*, 173–179. https://doi.org/10.1140/epjst/e2009-01026-6

Geier, M., Schönherr, M., Pasquali, A., & Krafczyk, M. (2015). The cumulant lattice Boltzmann equation in three dimensions: Theory and validation. *Computers & Mathematics with Applications, 70*(4), 507–547. https://doi.org/10.1016/j.camwa.2015.05.001

Gendre, F., Ricot, D., Fritz, G., & Sagaut, P. (2017). Grid refinement for aeroacoustics in the lattice Boltzmann method: A directional splitting approach. *Physical Review E, 96*, 023311. https://doi.org/10.1103/PhysRevE.96.023311

Guzik, S. M., Weisgraber, T. H., Colella, P., & Alder, B. J. (2014). Interpolation methods and the accuracy of lattice-Boltzmann mesh refinement. *Journal of Computational Physics, 259*, 461–487. https://doi.org/10.1016/j.jcp.2013.11.037


Hasert, M., Masilamani, K., Zimny, S., Klimach, H., Roller, S., & Bernsdorf, J. (2014). Complex fluid simulations with the parallel tree-based lattice Boltzmann solver Musubi. *Journal of Computational Science, 5*(5), 764–774. https://doi.org/10.1016/j.jocs.2013.11.001

Hsu, F.-S., Chang, K.-C., & Smith, M. R. (2018). Multi-block adaptive mesh refinement (AMR) for a lattice Boltzmann solver using GPUs. *Computers & Fluids, 175*, 48–52. https://doi.org/10.1016/j.compfluid.2018.01.033

Jaber, K., Essel, E. E., & Sullivan, P. E. (2025a). GPU-native adaptive mesh refinement with application to lattice Boltzmann simulations. *Computer Physics Communications*, 109543. https://doi.org/10.1016/j.cpc.2025.109543

Jaber, K., Essel, E. E., & Sullivan, P. E. (2025b). GPU-native embedding of complex geometries in adaptive octree grids applied to the lattice Boltzmann method. *arXiv preprint*, arXiv:2512.01251. https://arxiv.org/abs/2512.01251

Krüger, T., Kusumaatmaja, H., Kuzmin, A., Shardt, O., Silva, G., & Viggen, E. M. (2017). *The Lattice Boltzmann Method: Principles and Practice*. Springer.

Kutscher, K., Geier, M., & Krafczyk, M. (2018). Multiscale simulation of turbulent flow interacting with porous media based on a massively parallel implementation of the cumulant lattice Boltzmann method. *Computers & Fluids, 165*, 48–60. https://doi.org/10.1016/j.compfluid.2018.02.009

 *Revisiting grid refinement algorithms for the lattice Boltzmann method* (Doctoral thesis, University of Geneva). https://doi.org/10.13097/archive-ouverte/unige:26414

Lagrava, D., Malaspinas, O., Latt, J., & Chopard, B. (2012). Advances in multi-domain lattice Boltzmann grid refinement. *Journal of Computational Physics, 231*(14), 4808–4822. https://doi.org/10.1016/j.jcp.2012.03.015

Latt, J., Malaspinas, O., Kontaxakis, D., Parmigiani, A., Lagrava, D., Brogi, F., Belgacem, M. B., Thorimbert, Y., Leclaire, S., Li, S., Marson, F., Lemus, J., Kotsalos, C., Conradin, R., Coreixas, C., Petkantchin, R., Raynaud, F., Bény, J., & Chopard, B. (2021). Palabos: Parallel Lattice Boltzmann Solver. *Computers & Mathematics with Applications, 81*, 334–350. https://doi.org/10.1016/j.camwa.2020.03.022

Onodera, N., Idomura, Y., Ali, Y., & Shimokawabe, T. (2018a). Acceleration of wind simulation using locally mesh-refined lattice Boltzmann method on GPU-rich supercomputers. In *Lecture Notes in Computer Science* (Vol. 10777, pp. 128–145). Springer. https://doi.org/10.1007/978-3-319-69953-0_8

Onodera, N., Idomura, Y., Ali, Y., & Shimokawabe, T. (2018b). Communication reduced multi-time-step algorithm for real-time wind simulation on GPU-based supercomputers. In *Proceedings of the Workshop on Scalable Cluster Computing at SC18 (SCALA)*. IEEE. https://doi.org/10.1109/SCALA.2018.00005

Qi, J., Klimach, H., & Roller, S. (2019). Implementation of the compact interpolation within the octree based lattice Boltzmann solver Musubi. *Computers & Mathematics with Applications, 78*(4), 1131–1141. https://doi.org/10.1016/j.camwa.2016.06.025


Schönherr, M., Kucher, K., Geier, M., Stiebler, M., Freudiger, S., & Krafczyk, M. (2011). Multi-thread implementations of the lattice Boltzmann method on non-uniform grids for CPUs and GPUs. *Computers & Mathematics with Applications, 61*(12), 3730–3743. https://doi.org/10.1016/j.camwa.2011.04.012

Schönherr, M. (2015). *Towards reliable LES-CFD computations based on advanced LBM models utilizing (Multi-) GPGPU hardware* (Doctoral thesis, Technische Universität Braunschweig).

 Massively parallel algorithms for the lattice Boltzmann method on nonuniform grids. *SIAM Journal on Scientific Computing, 38*(4), C96–C126. https://doi.org/10.1137/15M1035240

Schornbaum, F., & Rüde, U. (2018). Extreme-scale block-structured adaptive mesh refinement. *SIAM Journal on Scientific Computing, 40*(3), C358–C387. https://doi.org/10.1137/17M1128411


Thorimbert, Y., Lagrava, D., Malaspinas, O., Chopard, B., Coreixas, C., de Santana Neto, J., Deiterding, R., & Latt, J. (2022). Local mesh refinement sensor for the lattice Boltzmann method. *Journal of Computational Science, 64*, 101864. https://doi.org/10.1016/j.jocs.2022.101864

Yu, Z., & Fan, L.-S. (2009). An interaction potential based lattice Boltzmann method with adaptive mesh refinement (AMR) for two-phase flow simulation. *Journal of Computational Physics, 228*(17), 6456–6478. https://doi.org/10.1016/j.jcp.2009.05.034

Zhang, W., et al. (2020). AMReX: a framework for block-structured adaptive mesh refinement. *Journal of Open Source Software, 5*(46), 1370. https://doi.org/10.21105/joss.01370
