# Power-Law Sudden Expansion Channel: Mathematical Specification

This document is the mathematical specification for the 2D symmetric sudden 3:1 expansion validation case with a power-law non-Newtonian fluid.
It is implemented in the `sim_nonnewtonian_expansion.cu` simulation.

---

## Problem Description

We validate the TNL-LBM power-law non-Newtonian implementation against the benchmark described by Manica & de Bortoli (2003) and the SimScale validation page. The case is a steady laminar flow of an incompressible power-law fluid through a planar channel that expands suddenly from height $H$ to $3H$.

* Geometry: 2D symmetric sudden expansion with expansion ratio $3:1$.
* Fluid model: power-law (Ostwald-de Waele) with indices $n = 0.5,\ 1.0,\ 1.5,\ 2.0$.
* Flow regime: laminar, $Re = 40$ for all cases.
* Inlet mean velocity: $V = 0.5\ \mathrm{m/s}$.
* Comparison location: fully developed profile at $x = 28\ \mathrm{m}$ in the expanded section.
* Spanwise setup: the domain is 3D with a 1 m physical spanwise width and periodic boundary conditions in $y$.

### References

* Manica, Rogerio and Álvaro L. de Bortoli. "Simulation of Incompressible Non-Newtonian Flows Through Channels with Sudden Expansion Using the Power-Law Model." Trends in Applied and Computational Mathematics 4 (2003): 333-340. https://tema.sbmac.org.br/tema/article/view/355/294
* SimScale. "Validation Case: Non-Newtonian Flow Through Expansion Channel." https://www.simscale.com/docs/validation-cases/non-newtonian-flow-through-expansion-channel/

---

## Domain Geometry

The channel is symmetric about the centerline $z = 0$. The expansion step is located at $x = 5\ \mathrm{m}$.

```text
                                     wall (GEO_WALL)
  z = +1.5                   +-------------------------------+
                             |                               |
                             |                               |
  z = +0.5  +----------------+                               |
            |  inlet section         expanded section        |
            |                            (fluid)             |
  z = -0.5  +----------------+                               |
                             |                               |
                             |                               |
  z = -1.5                   +-------------------------------+
            ^                ^                               ^
          x = 0            x = 5                         x = 30
                           (expansion step)
```

### Physical dimensions

| Region | Streamwise extent $x$ | Wall-normal extent $z$ | Half-height $R$ |
|--------|------------------------|-------------------------|-----------------|
| Inlet section | $[0, 5]\ \mathrm{m}$ | $[-0.5, 0.5]\ \mathrm{m}$ | $R_\mathrm{inlet} = 0.5\ \mathrm{m}$ |
| Expanded section | $[5, 30]\ \mathrm{m}$ | $[-1.5, 1.5]\ \mathrm{m}$ | $R_\mathrm{expanded} = 1.5\ \mathrm{m}$ |
| Comparison plane | $x = 28\ \mathrm{m}$ | fully developed expanded profile | $R_\mathrm{expanded}$ |

The spanwise direction $y$ uses a 1 m physical width. The expansion is symmetric: both the top and bottom walls step outward at $x = 5\ \mathrm{m}$, keeping the centerline at $z = 0$.

---

## Governing Equations

The flow is governed by the incompressible Navier-Stokes equations with a generalized Newtonian stress:

$$
\nabla \cdot \vec{u} = 0,
$$

$$
\rho \left( \frac{\partial \vec{u}}{\partial t} + \vec{u} \cdot \nabla \vec{u} \right) = -\nabla p + \nabla \cdot \boldsymbol{\tau}.
$$

For the power-law model the extra stress tensor is

$$
\boldsymbol{\tau} = 2 \, \mu_\mathrm{eff} \, \boldsymbol{D},
$$

where $\boldsymbol{D} = \frac{1}{2}\left( \nabla \vec{u} + (\nabla \vec{u})^\top \right)$ is the strain-rate tensor and the effective dynamic viscosity is

$$
\mu_\mathrm{eff} = K \, |\dot{\gamma}|^{n-1}.
$$

The scalar shear rate is

$$
\boxed{
\dot{\gamma} = \sqrt{2 \, \boldsymbol{D} : \boldsymbol{D}}.
}
$$

In kinematic form the viscosity is

$$
\nu(\dot{\gamma}) = \frac{K}{\rho} \, \dot{\gamma}^{\,n-1}.
$$

The generalized Reynolds number based on the inlet mean velocity $V$ and the inlet height $H$ is

$$
\boxed{
Re = \frac{\rho \, V^{2-n} \, H^n}{K}.
}
$$

---

## Analytical Solution

For fully developed plane Poiseuille flow in a channel of half-height $R$ driven by a constant pressure gradient $A = -\partial p / \partial x$, the power-law velocity profile is

$$
u(z) = \frac{n}{n+1} \left( \frac{A}{K} \right)^{1/n} \left( R^{(n+1)/n} - |z|^{(n+1)/n} \right), \qquad |z| \le R.
$$

The volumetric flow rate per unit span is

$$
Q = \frac{2n}{2n+1} R^2 \left( \frac{A R}{K} \right)^{1/n},
$$

so the mean velocity is

$$
\bar{u} = \frac{Q}{2R} = \frac{n}{2n+1} R \left( \frac{A R}{K} \right)^{1/n}.
$$

The centerline (maximum) velocity is therefore

$$
u_\mathrm{max} = \bar{u} \, \frac{2n+1}{n+1}.
$$

Expressing the profile in terms of the mean velocity gives the form used for boundary conditions and error evaluation:

$$
\boxed{
u(z) = u_\mathrm{max} \left( 1 - \left( \frac{|z|}{R} \right)^{1+1/n} \right), \qquad |z| \le R.
}
$$

This is the planar-slit form of Tanner's power-law channel solution. The pipe-flow form quoted in Manica & de Bortoli (their eqs. (3.1)-(3.2)) differs only in the centerline-to-mean ratio, $(3n+1)/(n+1)$ instead of $(2n+1)/(n+1)$, and is not used here. The Newtonian limit $n = 1$ recovers the parabola with $u_\mathrm{max} = 1.5\,\bar u$.

### Inlet profile

At the inlet $H_\mathrm{inlet} = 1\ \mathrm{m}$, $R_\mathrm{inlet} = 0.5\ \mathrm{m}$, and the prescribed mean velocity is $V = 0.5\ \mathrm{m/s}$. Hence

$$
\boxed{
u_\mathrm{max,inlet} = V \, \frac{2n+1}{n+1},
}
$$

and the inlet velocity profile is

$$
\boxed{
u_\mathrm{inlet}(z) = u_\mathrm{max,inlet} \left( 1 - \left( \frac{|z|}{R_\mathrm{inlet}} \right)^{1+1/n} \right), \qquad |z| \le R_\mathrm{inlet}.
}
$$

### Expanded profile at $x = 28\ \mathrm{m}$

Mass conservation gives the mean velocity in the expanded section:

$$
\boxed{
V_\mathrm{expanded} = V \, \frac{H_\mathrm{inlet}}{H_\mathrm{expanded}} = \frac{V}{3}.
}
$$

With $R_\mathrm{expanded} = 1.5\ \mathrm{m}$ the expanded-section centerline velocity is

$$
\boxed{
u_\mathrm{max,expanded} = V_\mathrm{expanded} \, \frac{2n+1}{n+1},
}
$$

and the analytical comparison profile is

$$
\boxed{
u_\mathrm{expanded}(z) = u_\mathrm{max,expanded} \left( 1 - \left( \frac{|z|}{R_\mathrm{expanded}} \right)^{1+1/n} \right), \qquad |z| \le R_\mathrm{expanded}.
}
$$

---

## Domain Discretization

The resolution level $N_\mathrm{res} \ge 1$ (integer) sets the lattice extents: $Z = 30\,N_\mathrm{res} + 4$ wall-normal nodes, $X = 300\,N_\mathrm{res} + 3$ streamwise nodes, and $Y = \max\!\bigl(4,\;\mathrm{round}(1.0 / \delta_x) + 2\bigr)$ spanwise nodes; the spacing is $\delta_x = 0.1/N_\mathrm{res}\ \mathrm{m}$. In the spanwise direction $y$, periodic ghost faces at $y = 0$ and $y = Y-1$ bracket the $Y - 2$ interior cells that span the 1 m physical width (periodic boundary conditions).

The following notes discuss the resulting discrete convention.

- **Halfway walls.** No-slip acts at faces halfway between wall nodes ($z = 1$, $z = Z-2$) and the adjacent fluid nodes: at resolution 1 the wall-node centers sit at $z_\mathrm{phys} = \mp 1.55\ \mathrm{m}$ while the effective walls are exactly at $\mp 1.5\ \mathrm{m}$. The same holds streamwise: the inlet ghost plane ($x = 0$) sits at $-\delta_x$ while the inflow BC node is exactly at $x_\mathrm{phys} = 0$, and the outlet node ($x = X-2$) sits exactly at $30\ \mathrm{m}$ while the ghost plane is at $30\ \mathrm{m} + \delta_x$.
- **Grid alignment.** With $\delta_x = 0.1/N_\mathrm{res}\ \mathrm{m}$ every resolution is fully aligned: $R_\mathrm{expanded\_lbm} = 15\,N_\mathrm{res}$ and $R_\mathrm{inlet\_lbm} = 5\,N_\mathrm{res}$ are always integers, the spanwise interior spans exactly 1 m ($10\,N_\mathrm{res}$ cells), and the comparison plane lands exactly at $x = 28\ \mathrm{m}$, and the outflow boundary node lies exactly at $x = 30\ \mathrm{m}$ (with $X = 300\,N_\mathrm{res} + 3$).

### Boundary conditions

The simulation uses the following TNL-LBM GEO tags.

| Boundary | Location | GEO tag | Condition |
|----------|----------|---------|-----------|
| Inlet ghost plane | lattice $x = 0$, $x_\mathrm{phys} = -\delta_x$ | `GEO_NOTHING` | Ghost layer |
| Inlet | $x_\mathrm{phys} = 0$ (lattice $x = 1$), $\|z_\mathrm{phys}\| \le 0.5\ \mathrm{m}$ | `GEO_INFLOW_LEFT` | Fixed power-law Poiseuille profile from Section 4 with $V = 0.5\ \mathrm{m/s}$; inflow BC height is 1 m |
| Inlet-section walls at inflow plane | $x_\mathrm{phys} = 0$ (lattice $x = 1$), $\|z_\mathrm{phys}\| > 0.5\ \mathrm{m}$ | `GEO_WALL` (interior, via `setMap`) | No-slip; carved on the inflow plane so the inflow BC is only 1 m tall |
| Outlet | lattice $x = X-2$, $x_\mathrm{phys} = 30\ \mathrm{m}$ | `GEO_OUTFLOW_RIGHT_INTERP` | Pressure outlet, reference density $\rho = 1$ |
| Outlet ghost plane | lattice $x = X-1$, $x_\mathrm{phys} = 30\ \mathrm{m} + \delta_x$ | `GEO_NOTHING` | Ghost layer |
| Top and bottom walls | wall faces at $\|z_\mathrm{phys}\| = 1.5\ \mathrm{m}$; wall nodes at lattice $z = 1$ and $z = Z-2$ | `GEO_WALL` | No-slip, $\vec{u} = 0$ |
| Expansion step | $0 \le x_\mathrm{phys} \le 5\ \mathrm{m}$ and $\|z_\mathrm{phys}\| > 0.5\ \mathrm{m}$ | `GEO_WALL` (interior, via `setMap`) | No-slip; includes the inflow plane at $x = 1$ |
| Wall-normal ghost planes | $z = 0$ and $z = Z-1$ | `GEO_NOTHING` | Ghost layers |
| Spanwise boundaries | $y = 0$ and $y = Y-1$ | `GEO_PERIODIC` | Periodic, 1 m physical width |

The inflow plane at lattice $x = 1$ has `GEO_INFLOW_LEFT` for $|z_\mathrm{phys}| \le 0.5\ \mathrm{m}$ (1 m height) and `GEO_WALL` for $|z_\mathrm{phys}| > 0.5\ \mathrm{m}$, so the effective inflow boundary condition matches the 1 m inlet-section height rather than the full 3 m expanded height.

The imposed inflow profile is the fully developed power-law profile (Manica & de Bortoli's "parabolic condition") rather than the uniform 0.5 m/s inlet used in the SimScale setup. Both impose the same volumetric flow rate (mean velocity 0.5 m/s); the developed profile is adopted because the comparison at $x = 28\ \mathrm{m}$ assumes fully developed flow and this choice avoids an artificial inlet-development transient.

---

### Lattice-to-physical mapping

The expanded-section physical height is $H_\mathrm{expanded} = 3\ \mathrm{m}$. With $Z$ lattice nodes in the wall-normal direction, the lattice spacing is

$$
\delta_x = \frac{H_\mathrm{expanded}}{Z - 4}.
$$

The wall-normal lattice index is centered on the channel centerline:

$$
\boxed{
z_\mathrm{center\_lbm} = \frac{Z-1}{2}, \qquad z_\mathrm{phys}(z) = \bigl(z - z_\mathrm{center\_lbm}\bigr) \, \delta_x.
}
$$

The no-slip walls act at the faces halfway between the wall nodes $z = 1$, $z = Z-2$ and the adjacent fluid nodes (halfway bounce-back): the effective wall positions are exactly $z_\mathrm{phys} = \mp 1.5\ \mathrm{m}$, while the wall-node centers sit one half-cell further out ($\mp 1.55\ \mathrm{m}$ at resolution 1). The fluid cells are $z = 2, \dots, Z-3$.

The streamwise index is offset by the inlet ghost layer: lattice $x = 1$ is the first physical cell at $x_\mathrm{phys} = 0$, so

$$
\boxed{
x_\mathrm{phys}(x) = (x - 1) \, \delta_x.
}
$$

Lattice $x = 0$ is the `GEO_NOTHING` ghost layer in front of the inlet, and $x = X-1$ is the ghost layer past the outlet.

In the code, both mappings are realized through the lattice coordinate origin `PHYS_ORIGIN`. It is assigned to the lattice's `physOrigin` member (`lat.physOrigin = PHYS_ORIGIN`), i.e. they are the same vector. TNL-LBM maps a lattice index $i$ to the physical point
$\mathrm{physOrigin} + (i - \tfrac{1}{2})\,\delta_x$
per vector component. The simulation sets

$$
\mathrm{PHYS\_ORIGIN} = \left(-\frac{\delta_x}{2},\; -\frac{\delta_x}{2},\; \left(0.5 - z_\mathrm{center\_lbm}\right)\,\delta_x\right),
$$

which reproduces the boxed formulas above: the first inflow plane lies at $x_\mathrm{phys} = 0$ (inlet ghost at $-\delta_x$), the centerline lies at $z_\mathrm{phys} = 0$, and the effective walls lie at $z_\mathrm{phys} = \mp 1.5\ \mathrm{m}$.

The half-heights in lattice units are

$$
\boxed{
R_\mathrm{inlet\_lbm} = \frac{R_\mathrm{inlet}}{\delta_x}, \qquad R_\mathrm{expanded\_lbm} = \frac{R_\mathrm{expanded}}{\delta_x}.
}
$$

For a resolution-1 grid ($Z = 34$), this gives $R_\mathrm{expanded\_lbm} = 15$ and $R_\mathrm{inlet\_lbm} = 5$.

---

## Case Parameters

All four cases share $\rho = 1\ \mathrm{kg/m^3}$, inlet mean velocity $V = 0.5\ \mathrm{m/s}$, inlet height $H = 1\ \mathrm{m}$, and $Re = 40$. The consistency index $K$ is adjusted per power-law index $n$.

| Case | $n$ | $K\ \mathrm{[Pa \cdot s^n]}$ | $V\ \mathrm{[m/s]}$ | $\rho\ \mathrm{[kg/m^3]}$ | $Re$ |
|:----:|:---:|:----------------------------:|:--------------------:|:--------------------------:|:----:|
| A | $0.5$ | $0.008838835$ | $0.5$ | $1$ | $40$ |
| B | $1.0$ | $0.0125$ | $0.5$ | $1$ | $40$ |
| C | $1.5$ | $0.01767767$ | $0.5$ | $1$ | $40$ |
| D | $2.0$ | $0.025$ | $0.5$ | $1$ | $40$ |

For each case the Reynolds number is verified by

$$
Re = \frac{\rho \, V^{2-n} \, H^n}{K} = 40.
$$

---

## Non-Dimensionalization and Unit Conversion

Diffusive scaling is based on the inlet section because the inlet has the highest velocities and therefore controls the Mach number.

Define the inlet centerline velocity

$$
u_\mathrm{max,inlet} = V \, \frac{2n+1}{n+1},
$$

the inlet half-height $R_\mathrm{inlet} = 0.5\ \mathrm{m}$, and the lattice spacing $\delta_x$ (`PHYS_DL` in the code).

The nominal inlet shear scale used for non-dimensionalization is

$$
\boxed{
\dot{\gamma}_\mathrm{ref} = \frac{u_\mathrm{max,inlet}}{R_\mathrm{inlet}}.
}
$$

It is nominal, not the true inlet-wall shear rate: for the power-law profile the wall value is $(1 + 1/n)\,\dot{\gamma}_\mathrm{ref}$ (a factor of 3 for $n = 0.5$ and 1.5 for $n = 2$). Only a self-consistent velocity-over-length scale is needed for the scaling below.

The corresponding reference kinematic viscosity is

$$
\boxed{
\nu_\mathrm{ref} = \frac{K}{\rho} \, \dot{\gamma}_\mathrm{ref}^{\,n-1}.
}
$$

Diffusive scaling relates the physical time step to the lattice viscosity:

$$
\boxed{
\delta_t = \frac{\nu_\mathrm{lbm} \, \delta_x^2}{\nu_\mathrm{ref}}.
}
$$

The lattice consistency index follows from dimensional consistency:

$$
\boxed{
K_\mathrm{lbm} = \frac{K}{\rho} \, \frac{\delta_t^{2-n}}{\delta_x^2}.
}
$$

The lattice viscosity at the reference shear rate is

$$
\boxed{
\nu_\mathrm{lbm} = K_\mathrm{lbm} \, \dot{\gamma}_\mathrm{ref,lbm}^{\,n-1},
}
$$

where $\dot{\gamma}_\mathrm{ref,lbm} = \dot{\gamma}_\mathrm{ref} \, \delta_t$. This $\nu_\mathrm{lbm}$ controls the collision frequency $\omega$ and is invariant across resolutions. For $n = 2$ the exponent $2-n$ vanishes, so $K_\mathrm{lbm} = (K/\rho) / \delta_x^2$.

---

## Mach Number Stability Analysis

The lattice speed of sound is

$$
c_s = \frac{1}{\sqrt{3}} \approx 0.5774.
$$

The lattice Mach number is based on the inlet centerline velocity:

$$
\boxed{
Ma = \frac{u_\mathrm{max,inlet,lbm}}{c_s}, \qquad u_\mathrm{max,inlet,lbm} = u_\mathrm{max,inlet} \, \frac{\delta_t}{\delta_x}.
}
$$

The collision frequency is

$$
\omega = \frac{1}{3 \nu_\mathrm{lbm} + 0.5}.
$$

The lattice viscosity $\nu_\mathrm{lbm}$ is a free input (the `--lbm-viscosity` CLI option), not a derived quantity. In this specification it is chosen per power-law index so that the inlet Mach number is $0.10$; from $Ma = u_\mathrm{max,inlet}\,\nu_\mathrm{lbm}\,\delta_x/(\nu_\mathrm{ref}\,c_s)$ the required value is

$$
\nu_\mathrm{lbm} = \frac{Ma\, c_s\, \nu_\mathrm{ref}}{u_\mathrm{max,inlet}\,\delta_x}.
$$

At resolution $1$ ($Z = 34$ lattice nodes in the wall-normal direction, $\delta_x = 0.1\ \mathrm{m}$), the per-case stability data are:

| $n$ | $u_\mathrm{max,inlet}\ \mathrm{[m/s]}$ | $\dot{\gamma}_\mathrm{ref}\ \mathrm{[1/s]}$ | $\nu_\mathrm{ref}\ \mathrm{[m^2/s]}$ | $\nu_\mathrm{lbm}$ | $\omega$ | $Ma$ |
|:---:|:-------------------------------------:|:-------------------------------------------:|:-------------------------------------:|:------------------:|:--------:|:----:|
| $0.5$ | $0.667$ | $1.333$ | $0.00766$ | $0.0066$ | $1.923$ | $0.10$ |
| $1.0$ | $0.750$ | $1.500$ | $0.01250$ | $0.0096$ | $1.891$ | $0.10$ |
| $1.5$ | $0.800$ | $1.600$ | $0.02236$ | $0.0161$ | $1.823$ | $0.10$ |
| $2.0$ | $0.833$ | $1.667$ | $0.04167$ | $0.0289$ | $1.705$ | $0.10$ |

All values give $Ma \approx 0.10 < 0.3$ and $\omega < 2$, which is stable for the cumulant collision operator ($\omega$ is evaluated at the unrounded $\nu_\mathrm{lbm}$; recomputing from the printed 2-3-digit values shifts the last decimal). Choosing the viscosity freely instead (for example the code default `--lbm-viscosity 0.05`) would give $Ma \approx 0.75$ ($n = 0.5$), $0.52$ ($n = 1$) and $0.31$ ($n = 1.5$) at resolution 1, past the 0.3 comfort limit in all three cases; this is why the per-case tuning above is part of the specification rather than an implementation detail.

---

## Implementation Notes

This section collects the modeling deviations identified when reviewing this specification against the implementation. None of them change the specification itself, but they matter when interpreting the comparison to analytical solution or reference results.

### Split-viscosity forcing and shear-rate regularization

The solver does not collide with the full local viscosity. A freely chosen background viscosity $\nu_0$ (`--lbm-viscosity`) enters the collision operator; the remainder is applied as an explicit body force. In the momentum equation the viscous force is $\vec{f} = \nabla\cdot\boldsymbol{\tau}$ with the extra stress $\boldsymbol{\tau} = 2\,\nu(\dot\gamma)\,\boldsymbol{D}$, and it is split as

$$
\vec{f} = \underbrace{\nabla\cdot\bigl(2\,\nu_0\,\boldsymbol{D}\bigr)}_{\text{collision}} \;+\; \underbrace{\nabla\cdot\bigl(2\,\bigl(\nu(\dot\gamma) - \nu_0\bigr)\,\boldsymbol{D}\bigr)}_{\text{non-Newtonian forcing}}.
$$

The forcing is evaluated via the product rule, $(\nu - \nu_0)\,\nabla\cdot\boldsymbol{D} + \nabla(\nu - \nu_0)\cdot\boldsymbol{D}$, which reduces to $(\nu - \nu_0)\,\nabla\cdot\boldsymbol{D} + \nabla\nu\cdot\boldsymbol{D}$ since $\nu_0$ is constant, so the $\nabla\nu\cdot\boldsymbol{D}$ term is captured explicitly (see `include/lbm3d/nonNewtonian.h`). The power law itself is smoothed by a $C^\infty$ regularization that avoids a kink in $\nabla\nu$ while staying exact above single-precision noise; in lattice units the smooth floor is reached at $\dot\gamma \sim 10^{-5}$. SimScale's benchmark similarly clips $\nu$ to $[10^{-6}, 0.5]\ \mathrm{m^2/s}$. Around $x = 28\ \mathrm{m}$ the regularized region ($\dot\gamma/\dot\gamma_\mathrm{ref} \lesssim 10^{-3}$) is confined to a thin layer near the centerline where the velocity profile is flat, so its influence on the L1/L2 comparison is negligible.

### Run control

Runs use the D3Q27 lattice with the cumulant collision operator and the AB streaming pattern. Termination uses a probe-based convergence criterion (a moving window on the L1 error against the analytical profile) with a hard limit `physFinalTime` $= 10\,t_\mathrm{steady}$. The estimate $t_\mathrm{steady}$ is built from the *inlet* reference viscosity, whereas the slowest diffusion happens in the expanded section; for shear-thickening models ($n > 1$) the expanded-section diffusion time exceeds the inlet-based estimate by roughly an order of magnitude, leaving only a slim (about $1.1\times$) margin before the hard limit fires. If a shear-thickening run reports `flag.finished` instead of `flag.terminated`, base $t_\mathrm{steady}$ on the expanded-section viscosity or raise the limit.
