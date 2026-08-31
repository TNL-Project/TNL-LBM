# Moment-based boundary conditions for D2Q9 and D3Q27

## Reference

P. Eichler, R. Fučík, P. Strachota,
"Investigation of mesoscopic boundary conditions for lattice Boltzmann method in laminar flow problems",
*Computers & Mathematics with Applications* **173**, 87–101 (2024).
https://doi.org/10.1016/j.camwa.2024.08.009

The methodology follows the moment-based framework of T. Reis, T. Phillips,
"Moment-based boundary conditions for lattice Boltzmann simulations of diffusion in microfluidic channels" and its 3D extension
by I. Krastins, A. Kao, K. Pericleous, T. Reis,
"Moment-based boundary conditions for straight on-grid boundaries in three-dimensional lattice Boltzmann simulations",
*International Journal for Numerical Methods in Fluids* **92**, 2151–2165 (2020).
https://doi.org/10.1002/fld.4856

## Principle

At a boundary face, certain populations are unknown (those that would stream in from outside the domain).
The moment-based method reconstructs them by imposing constraints on a matching number of linearly independent hydrodynamic
moments.
For a **velocity boundary** (inflow), the imposed quantities are the velocity components;
density is derived from mass conservation.

The moment constraints are set to their **equilibrium values** (Chapman–Enskog zeroth-order approximation),
neglecting non-equilibrium corrections at the boundary.
This is justified by the fact that the equilibrium moment values are the physically consistent closure for the NSE at leading
order.

## D2Q9 velocity inflow boundary (left face, x = 0)

### Unknown populations

At the left boundary face,
the populations pointing in the +x direction are unknown after streaming (they would arrive from outside the domain):

| Direction | Symbol | c_x | c_y |
|-----------|--------|-----|-----|
| +x        | `pz`   | +1  |  0  |
| +x,+y     | `pp`   | +1  | +1  |
| +x,−y     | `pm`   | +1  | −1  |

The 6 known populations are: `zz` (0,0), `mz` (−1,0), `zp` (0,+1), `zm` (0,−1), `mm` (−1,−1), `mp` (−1,+1).

### Moment constraints (3 equations for 3 unknowns)

Following the Eichler/Reis methodology, the 3 constraints are:

1. **x-momentum** ($m_{10} = \rho \, v_x$): used to derive $\rho$, since $v_x$ is imposed.
2. **y-momentum** ($m_{01} = \rho \, v_y$): tangential momentum.
3. **Normal stress** ($m_{02} = \Pi_{yy} = \rho/3 + \rho \, v_y^2$): 2nd-order moment in the tangential direction,
   evaluated at equilibrium.

#### Constraint 1: density from mass + x-momentum

$$\rho = \sum_i f_i$$

x-momentum:

$$\rho \, v_x = \sum_i c_{i,x} f_i = (f_{pz} - f_{mz}) + (f_{pp} - f_{mm}) + (f_{pm} - f_{mp})$$

Combining (known sum $S = f_{zz} + f_{mz} + f_{zp} + f_{zm} + f_{mm} + f_{mp}$, unknown sum $U = f_{pz} + f_{pp} + f_{pm}$):

$$\rho = S + U, \qquad \rho \, v_x = U - (f_{mz} + f_{mm} + f_{mp})$$

Eliminating U:

$$\rho = \frac{f_{zz} + f_{zp} + f_{zm} + 2(f_{mz} + f_{mm} + f_{mp})}{1 - v_x}$$

#### Constraint 2: y-momentum

$$m_{01} = \sum_i c_{i,y} f_i = (f_{zp} - f_{zm}) + (f_{pp} - f_{mm}) + (f_{mp} - f_{pm}) = \rho \, v_y$$

Solving for the unknown combination:

$$f_{pp} - f_{pm} = \rho \, v_y - (f_{zp} - f_{zm}) + (f_{mm} - f_{mp})$$

#### Constraint 3: normal stress $\Pi_{yy}$ (equilibrium)

$$m_{02} = \sum_i c_{i,y}^2 \, f_i = (f_{zp} + f_{zm}) + (f_{pp} + f_{pm}) + (f_{mp} + f_{mm}) = \frac{\rho}{3} + \rho \, v_y^2$$

Solving for the unknown combination:

$$f_{pp} + f_{pm} = \frac{\rho}{3} + \rho \, v_y^2 - f_{zp} - f_{zm} - f_{mp} - f_{mm}$$

### Reconstruction formulas

Adding and subtracting constraints 2 and 3:

$$f_{pp} = \frac{1}{2}\left(\frac{\rho}{3} + \rho \, v_y^2 + \rho \, v_y - 2 f_{zp} - 2 f_{mp}\right)$$

$$f_{pm} = \frac{1}{2}\left(\frac{\rho}{3} + \rho \, v_y^2 - \rho \, v_y - 2 f_{zm} - 2 f_{mm}\right)$$

$$f_{pz} = \rho - f_{zz} - f_{mz} - f_{zp} - f_{zm} - f_{pp} - f_{pm} - f_{mm} - f_{mp}$$

### Equilibrium verification

At equilibrium ($v_x = 0$, $v_y = 0$, $\rho = 1$), the D2Q9 weights are:

$$f_{zz} = 4/9, \qquad f_{pz} = f_{mz} = f_{zp} = f_{zm} = 1/9, \qquad f_{pp} = f_{mm} = f_{pm} = f_{mp} = 1/36$$

- $\rho$: $(4/9 + 1/9 + 1/9 + 2(1/9 + 1/36 + 1/36)) / 1 = 6/9 + 2(6/36) = 2/3 + 1/3 = 1$ ✓
- $f_{pp}$: $(1/3 + 0 + 0 - 2/9 - 2/36) / 2 = (12/36 - 8/36 - 2/36) / 2 = 1/36$ ✓
- $f_{pm}$: $(1/3 + 0 - 0 - 2/9 - 2/36) / 2 = 1/36$ ✓
- $f_{pz}$: $1 - 4/9 - 1/9 - 1/9 - 1/9 - 1/36 - 1/36 - 1/36 - 1/36 = 1 - 8/9 = 1/9$ ✓

### Conservation verification

- **Mass**: $f_{pz} + f_{pp} + f_{pm} = \rho - S$ ✓ (by construction of $f_{pz}$)
- **x-momentum**: $f_{pz} + f_{pp} + f_{pm} - (f_{mz} + f_{mm} + f_{mp}) = \rho \, v_x$ ✓ (from constraint 1)
- **y-momentum**: $(f_{zp} - f_{zm}) + (f_{pp} - f_{mm}) + (f_{mp} - f_{pm}) = \rho \, v_y$ ✓ (from constraint 2)
- **$\Pi_{yy}$**: $(f_{zp} + f_{zm}) + (f_{pp} + f_{pm}) + (f_{mp} + f_{mm}) = \rho/3 + \rho \, v_y^2$ ✓ (from constraint 3)

## D3Q27 velocity inflow boundary (left face, x = 0)

### Unknown populations (9)

All +x directions: `pzz`, `ppz`, `pmz`, `pzp`, `pzm`, `ppp`, `ppm`, `pmp`, `pmm`.

### Moment constraints (9 equations for 9 unknowns)

The Eichler formulation for D3Q27 uses moments up to 4th order, all set to equilibrium:

| Moment | Symbol | Equilibrium value | Role |
|--------|--------|-------------------|------|
| $m_{100}$ | $\rho \, v_x$ | imposed $v_x$ → derives $\rho$ | normal momentum |
| $m_{010}$ | $\rho \, v_y$ | imposed $v_y$ | tangential momentum 1 |
| $m_{001}$ | $\rho \, v_z$ | imposed $v_z$ | tangential momentum 2 |
| $m_{020}$ ($\Pi_{yy}$) | $\rho/3 + \rho \, v_y^2$ | equilibrium | normal stress $y$ |
| $m_{002}$ ($\Pi_{zz}$) | $\rho/3 + \rho \, v_z^2$ | equilibrium | normal stress $z$ |
| $m_{011}$ ($\Pi_{yz}$) | $\rho \, v_y v_z$ | equilibrium | shear stress $yz$ |
| $m_{021}$ ($Q_{yyz}$) | $\rho \, v_z/3 + \rho \, v_y^2 v_z$ | equilibrium | 3rd-order cross |
| $m_{012}$ ($Q_{yzz}$) | $\rho \, v_y/3 + \rho \, v_y v_z^2$ | equilibrium | 3rd-order cross |
| $m_{022}$ ($S_{yyzz}$) | $\rho/9 + \rho/3 \, (v_y^2 + v_z^2) + \rho \, v_y^2 v_z^2$ | equilibrium | 4th-order cross |

The first constraint (m₁₀₀) derives ρ from the known populations and imposed v_x:

$$\rho = \frac{F_Z + 2 F_W}{1 - v_x}, \qquad F_Z = f_{zzz} + \sum_{\substack{c_x = 0 \\ \text{off-axis}}} f_i, \qquad F_W = f_{mzz} + \sum_{\substack{c_x = -1 \\ \text{off-axis}}} f_i$$

The remaining 8 constraints reconstruct the 8 unknown diagonal/edge DFs with closed per-slot formulas (listed in the
verification section below), symmetrized so that mirror directions are grouped into commutative pairs.
The `pzz` (axis-aligned +x) DF is reconstructed last from mass conservation.

## Verification against implementation

The legacy XM implementations are preserved below in full as the bitwise reference for the generalization:
the current generalized bodies (see below) reproduce their computed values bit-exactly on the legacy face, by construction.

### D2Q9

The 2D implementation is a faithful reduction of the Eichler moment BC using mass + y-momentum + $\Pi_{yy}$ (normal stress).
Full legacy XM body (`GEO_INFLOW_LEFT`, replaced by the `<0,-1>` instantiation of the templated body in the generalization):

```cpp
KS.rho = (KS.f[zz] + (KS.f[zp] + KS.f[zm]) + 2 * (KS.f[mz] + (KS.f[mm] + KS.f[mp]))) / (1 - KS.vx);
dreal m01 = KS.rho * KS.vy;
dreal m02 = n1o3 * KS.rho + KS.rho * (KS.vy * KS.vy);
KS.f[pp] = (dreal) 0.5 * (m02 + m01 - 2 * KS.f[zp] - 2 * KS.f[mp]);
KS.f[pm] = (dreal) 0.5 * (m02 - m01 - 2 * KS.f[zm] - 2 * KS.f[mm]);
KS.f[pz] = KS.rho - KS.f[zz] - KS.f[mz] - (KS.f[zp] + KS.f[zm]) - (KS.f[pp] + KS.f[pm]) - (KS.f[mm] + KS.f[mp]);
```

**Match: ✓** — the `<0,-1>` instantiation reproduces every assignment: Z pair positive-first `(zp + zm)`,
W pair negative-first `(mm + mp)`, corners first, and the axis slot closing the mass budget in the legacy subtraction order.

### D3Q27

The 3D implementation matches the Eichler formulation with the 9 equilibrium moments, using the symmetrized expressions below.
Full legacy XM body (`GEO_INFLOW_LEFT`, replaced by the runtime-parameterized body called on the detected face):

```cpp
KS.rho = (dreal) 1.0 / (1 - KS.vx) *
        (
            (KS.f[zzz] + (
                + ((KS.f[zpp] + KS.f[zmm]) + (KS.f[zpm] + KS.f[zmp]))
                + ((KS.f[zpz] + KS.f[zmz]) + (KS.f[zzp] + KS.f[zzm]))
            ))
            + 2 * (KS.f[mzz] + (
                + ((KS.f[mpp] + KS.f[mmm]) + (KS.f[mpm] + KS.f[mmp]))
                + ((KS.f[mpz] + KS.f[mmz]) + (KS.f[mzp] + KS.f[mzm]))
            ))
        );
dreal m100 = KS.rho * KS.vx;
dreal m010 = KS.rho * KS.vy;
dreal m001 = KS.rho * KS.vz;
dreal m011 = KS.rho * (KS.vy * KS.vz);
dreal m020 = n1o3 * KS.rho + KS.rho * (KS.vy * KS.vy);
dreal m002 = n1o3 * KS.rho + KS.rho * (KS.vz * KS.vz);
dreal m021 = n1o3 * KS.rho * KS.vz + KS.rho * ((KS.vy * KS.vy) * KS.vz);
dreal m012 = n1o3 * KS.rho * KS.vy + KS.rho * (KS.vy * (KS.vz * KS.vz));
dreal m022 = n1o9 * KS.rho + n1o3 * KS.rho * (KS.vy * KS.vy + KS.vz * KS.vz) + KS.rho * (KS.vy * KS.vy) * (KS.vz * KS.vz);
KS.f[pzz] = m100 + (m022 - (m020 + m002))
    + KS.f[mzz]
    + (((KS.f[zpp] + KS.f[zmm]) + (KS.f[zpm] + KS.f[zmp])) + ((KS.f[zzp] + KS.f[zzm]) + (KS.f[zpz] + KS.f[zmz])))
    + 2 * (((KS.f[mpp] + KS.f[mmm]) + (KS.f[mpm] + KS.f[mmp])) + ((KS.f[mpz] + KS.f[mmz]) + (KS.f[mzp] + KS.f[mzm])));
KS.f[ppz] = (dreal) 0.5 * ((m020 - m022) + (-m012 + m010)) - (KS.f[mpz] + KS.f[zpz]);
KS.f[pmz] = (dreal) 0.5 * ((m020 - m022) + (m012 - m010)) - (KS.f[mmz] + KS.f[zmz]);
KS.f[pzp] = (dreal) 0.5 * ((m002 - m022) + (-m021 + m001)) - (KS.f[mzp] + KS.f[zzp]);
KS.f[pzm] = (dreal) 0.5 * ((m002 - m022) + (m021 - m001)) - (KS.f[mzm] + KS.f[zzm]);
KS.f[ppp] = (dreal) 0.25 * ((m022 + m011) + (m021 + m012)) - (KS.f[mpp] + KS.f[zpp]);
KS.f[ppm] = (dreal) 0.25 * ((m022 - m011) + (-m021 + m012)) - (KS.f[mpm] + KS.f[zpm]);
KS.f[pmp] = (dreal) 0.25 * ((m022 - m011) + (m021 - m012)) - (KS.f[mmp] + KS.f[zmp]);
KS.f[pmm] = (dreal) 0.25 * ((m022 + m011) + (-m021 - m012)) - (KS.f[mmm] + KS.f[zmm]);
```

**Match: ✓** — the runtime body on XM reproduces every assignment;
note the rebuilt axis slot `pzz` swaps the order of the two tangential axis pairs relative to the density
(`(zzp + zzm) + (zpz + zmz)` above), which is bitwise-invisible because the IEEE add is commutative,
so the generalized body keeps a single sum per layer for both uses.

The conditioning principle behind the symmetrized forms in both models —
pairing mirror directions and reordering expressions to suppress float32 round-off —
follows Appendix J ("Well conditioned collision operator") of M. Geier, M. Schönherr, A. Pasquali, M. Krafczyk,
"The cumulant lattice Boltzmann equation in three dimensions: Theory and validation,"
*Computers & Mathematics with Applications* **70**, 507–547 (2015), https://doi.org/10.1016/j.camwa.2015.05.001

## Generalization to all faces (`GEO_INFLOW_MOMENT`)

The legacy case (`GEO_INFLOW_LEFT`) is the face with outward normal **−x** at the domain's low-x wall.
The implementation generalizes it to any of the six (D3Q27) or four (D2Q9) domain faces without rotating coordinates:
the face enters the reconstruction formulas directly.

### Face parameters

| Face | outward normal n | normal axis a | unknown (written) directions: component along a equal to | known moving layer W (outward): component equal to | tangential axes (t1, t2) |
|------|------------------|---------------|----------------------------------------------------------|----------------------------------------------------|---------------------------|
| XP | +x | 0 | −1 | +1 | (y, z) |
| XM | −x (legacy) | 0 | +1 | −1 | (y, z) |
| YP | +y | 1 | −1 | +1 | (z, x) |
| YM | −y | 1 | +1 | −1 | (z, x) |
| ZP | +z | 2 | −1 | +1 | (x, y) |
| ZM | −z | 2 | +1 | −1 | (x, y) |

Tangential axes are cyclic: after the normal axis, t1 and t2 are the remaining two in (x,y,z) order.
The **Z layer** holds all directions with normal component 0.

Let $s$ be the outward sign ($\pm 1$), $v_n$ the velocity component along the normal axis,
$v_{t1}$ and $v_{t2}$ the tangential ones.
The unknown populations are exactly those with normal component $-s$ (moving into the domain);
in the code this is the per-direction test `dcomp(i, AXIS) == -SIGN`.

### General formulas

Density (the twice-weighted layer is the *outward-moving* one, W; the Z-layer axis slot always includes `f[zzz]` / `f[zz]`):

$$\rho = \frac{F_Z + 2 F_W}{1 + s \, v_n}$$

Here $F_Z$ includes the axis slot; below,
$F'_Z$/$F'_W$ denote the same layer sums without the axis slot (the code's `zSum`/`wSum`).

Check (XM): $s = -1$, $W = \{c_x = -1\}$ gives the legacy $(F_Z + 2 F_W)/(1 - v_x)$ ✓.

Moments are evaluated with the face-local tangential velocity components (same equilibrium shapes as the legacy case with
$v_y \to v_{t1}$ and $v_z \to v_{t2}$: $m_{010} \to \rho \, v_{t1}$, $m_{011} \to \rho \, v_{t1} v_{t2}$,
$m_{020} \to \rho/3 + \rho \, v_{t1}^2$, $m_{021} \to \rho \, v_{t2}/3 + \rho \, v_{t1}^2 v_{t2}$,
$m_{012} \to \rho \, v_{t1}/3 + \rho \, v_{t1} v_{t2}^2$,
$m_{022} \to \rho/9 + \rho/3 \, (v_{t1}^2 + v_{t2}^2) + \rho \, v_{t1}^2 v_{t2}^2$; the normal momentum is $m_N = \rho \, v_n$).

The formulas below use the source-code moment symbols: `mN` (normal momentum $\rho \, v_n$),
`mT1`/`mT2` (tangential momenta $\rho v_{t1}$, $\rho v_{t2}$), `mT1T1`/`mT2T2` (tangential stresses $\rho/3 + \rho v_{t1}^2$,
$\rho/3 + \rho v_{t2}^2$), `mT1T2` (the shear $\rho v_{t1} v_{t2}$), `mT1T1T2`/`mT1T2T2` (the 3rd-order cross moments),
and `mTT` (the 4th-order tangential stress) —
the face-local analogs of the Eichler indices $m_{010}, m_{020}, m_{011}, m_{021}, m_{012}, m_{022}$ above.

Reconstruction of an unknown direction with tangential components (ct1, ct2) ∈ {−1,0,+1}² —
written slot gets the moment expression minus the same-tangential slots of the other two layers Z and W (partner slots
`f[W(ct1,ct2)]` and `f[Z(ct1,ct2)]`, always untouched, so write order is free):

- **axis (0,0)**: `−s mN + (mTT − (mT1T1 + mT2T2)) + f[axis-W] + F'_Z + 2 F'_W`
- **t1-edge (±1,0)**: `0.5 · ((mT1T1 − mTT) + ct1·(mT1 − mT1T2T2)) − (f[W] + f[Z])`
- **t2-edge (0,±1)**: `0.5 · ((mT2T2 − mTT) + ct2·(mT2 − mT1T1T2)) − (f[W] + f[Z])`
- **corner (±1,±1)**: `0.25 · ((mTT + ct1·ct2·mT1T2) + (ct2·mT1T1T2 + ct1·mT1T2T2)) − (f[W] + f[Z])`

D2Q9 analog with the 3-constraint system (mass, tangential momentum $m_{01} = \rho \, v_t$,
tangential stress $m_{02} = \rho/3 + \rho \, v_t^2$): corners $0.5 \cdot (m_{02} + ct \cdot m_{01} - 2 f[Z] - 2 f[W])$,
axis `rho − Σ(all other slots)` written last (it reads the just-written corners).

Instantiating XM recovers the legacy assignment list exactly (up to summation grouping); see the reconstruction lists above.

### Bitwise compatibility and dispatch

There is one source body per model for every face:
D3Q27's runtime-parameterized `inflowMoment(face, KS)` and D2Q9's template `inflowMoment<AXIS, SIGN>(KS)`.
The `GEO_INFLOW_MOMENT` case detects the face from the map at runtime, and the two models then dispatch it differently,
as measured in the fused production kernels:

- **D3Q27** calls the runtime body directly in both streaming patterns —
  the D3Q27 kernels tolerate the runtime face (and the unified call is the fastest shape measured there).
- **D2Q9** carries one `template <int AXIS, int SIGN>` body whose slot arithmetic (`T`, the layer slots,
  the written-layer slots) is all `constexpr`, and the pre-collision switch instantiates it per face (`<0,-1>` XM,
  `<0,+1>` XP, `<1,+1>` YP, `<1,-1>` YM):
  a shared runtime-face body loses this folding and was measured worse in D2Q9's small fused kernels (hills A-A ~5%,
  register/literal pressure) and, unlike the folded instantiations,
  drifts production values away from the legacy BC (the hills mass-conservation regression check fails at final time) because
  the production kernel's FP-contraction context differs from the test kernels the pins were verified in.

The pre-generalization XM body is no longer carried in source;
bitwise compatibility with legacy simulations (inflow at a −x-normal face) follows from the construction below and is enforced
by the production regression suites (`test_d2q9.py`, `test_d3q27_nse.py -k sim_1`).
The remaining faces have no legacy bitwise requirement; their closed forms were validated independently:

- scripted symbolic/numerical check (same constraints as below) over randomized populations;
- unit suite `test_inflow_moment.cu` (`inflowmoment2d`/`inflowmoment3d`: all 4/6 faces,
  with constraints taken from the moment system itself — mass, momenta, tangential stresses;
  the 3D suite additionally shear and the 3rd/4th-order cross moments `q112`/`q122`/`m22`; written layer exactly `cn == -s`;
  untouched slots bit-exact), 0 failures in A-B and A-A builds.

### Balanced-pairing construction (bitwise isomorphism on the legacy face)

The generalized bodies reproduce the *computed values* of the pre-generalization XM body bit-exactly on the legacy face (D3Q27's
runtime body, and D2Q9's `<0,-1>` instantiation;
established during development by unit suites comparing against the since-removed verbatim specialization:
0 differing slots over 2000 randomized inputs in both streaming patterns;
ongoing compatibility is gated by the production regression suites),
by constructing equal roundings at every point where a plain runtime-parameterized formula would diverge:

- D3Q27 layer sums group the corner mirror pairs
  `((+1,+1)+(−1,−1)) + ((+1,−1)+(−1,+1))` first, then the tangential edge pairs `((+1,0)+(−1,0)) + ((0,+1)+(0,−1))`.
  The operand order *inside* each pair —
  and the legacy body's t1/t2 pair-order swap of the Z axis block between the density and the rebuilt axis slot —
  is bitwise-invisible because the IEEE add is commutative, so a single sum per layer serves all uses.
- The 3D density is formed as $1/(1 + s \, v_n)$ (reciprocal first) times the layer sum —
  dividing the full sum instead changes the last bit.
- The remaining danger is the compiler's `fp-contract=fast`:
  in the legacy body NVVM fuses several non-exact products into their consuming adds (e.g.
  `n1o3*rho` scaled by `v_t2`/`v_t1` fused into `m021`/`m012`, `rho*v_t2^2` fused into `m002`, `rho*v_n` into the axis slot's
  leading term, `rho*(v_t1*v_t2)` into each corner write, and the `m022` chain),
  but which mul feeds which fma depends on the whole-function data-flow graph and cannot be reproduced reliably from a
  runtime-parameterized source.
  The generic body therefore pins exactly those contraction spots with `lbm_fma_rn` (which never re-contracts and has a
  canonical operand order) — 7 call sites in D3Q27: `mT2T2`, `mT1T1T2`, `mT1T2T2`, two inside `mTT`, the axis slot,
  and the corner — while leaving legacy-plain operations (multiplicative factors `±1`, `2`, `0.5`, `0.25`,
  which are exact and contract losslessly, and the `m020`/`m01x` chains) untouched.
- D2Q9 sums each layer's `ct == ±1` pair before joining the axis slot (negative direction first inside the W pair),
  divides the density by $(1 + s \, v_n)$ at the end,
  pins `n1o3*rho` fused into the tangential stress and `rho*v_t` fused into each corner write (2 call sites producing 3 pinned
  evaluations: the stress and one per corner, with the plain tangential momentum folded into the corner pin),
  and forms the axis slot as the legacy subtraction chain
  `rho − f[Z-axis] − f[W-axis] − (Z pair) − (written corners) − (W pair)`,
  written last because it reads the just-written corner values.

Nothing in the reconstruction depends on a particular face:
the pairing and the pins are evaluated through the `AXIS`/`T1`/`T2`/`SIGN` indices — runtime values in D3Q27's single body,
`constexpr` from the `<AXIS, SIGN>` template parameters in D2Q9 —
so one source body per model serves all faces (the A-A fused-kernel spill lesson stands).
Whether production calls it directly (D3Q27) or instantiates it per face (D2Q9) is a dispatch property,
not a second implementation.

## Stability note

The moment-based BC uses equilibrium approximations for the higher-order moments ($\Pi_{yy}$ in 2D; $\Pi_{yy}$, $\Pi_{zz}$,
$\Pi_{yz}$, $Q_{yyz}$, $Q_{yzz}$, $S_{yyzz}$ in 3D).
The non-equilibrium corrections (proportional to viscosity τ) are neglected, which is the standard leading-order closure.
This approximation is valid for low-Mach, near-equilibrium flows and introduces O(τ) errors at the boundary,
consistent with the second-order accuracy of the LBM.
