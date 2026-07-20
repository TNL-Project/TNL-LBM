# Exact Analytical Solutions for Non-Newtonian Hagen–Poiseuille Flow

**Models:** Power-law, Carreau–Yasuda (CY), and Casson
**Geometry:** Plane Poiseuille (between parallel plates) and circular pipe
**Reference:** Power-law derivation from [course notes](https://jlk.fjfi.cvut.cz/md/4I8g_Vp1Rhmto8qmYk3AwQ?view), §4.5B
**Verification:** Mathematical precision checked by Oracle reasoning agent

---

## Table of Contents

1. [Problem Setup and General Framework](#1-problem-setup-and-general-framework)
2. [General Solution for Arbitrary $\eta(\dot\gamma)$](#2-general-solution-for-arbitrary-etadotgamma)
3. [Power-Law Model](#3-power-law-model)
4. [Carreau–Yasuda (CY) Model](#4-carreauyasuda-cy-model)
5. [Casson Model](#5-casson-model)
6. [Pipe (Circular Cross-Section) Solutions](#6-pipe-circular-cross-section-solutions)
7. [Volumetric Flow Rate](#7-volumetric-flow-rate)
8. [Limiting Cases and Verification](#8-limiting-cases-and-verification)
9. [Relationship to the TNL-LBM Code](#9-relationship-to-the-tnl-lbm-code)
* [Appendix: Hypergeometric Identity](#appendix-hypergeometric-identity)

---

## 1. Problem Setup and General Framework

### Assumptions

- Steady, laminar, incompressible flow ($\rho = \text{const}$)
- No external body forces
- Unidirectional flow: $\vec{V} = (W(x_3),\; 0,\; 0)^\top$
- No-slip at walls: $W(\pm R) = 0$
- Pressure depends only on $x_1$: $P = P(x_1)$, with constant gradient

$$\frac{\partial P}{\partial x_1} = \frac{P_\text{out} - P_\text{in}}{L} = -A, \qquad A \equiv \frac{P_\text{in} - P_\text{out}}{L} > 0$$

### Shear Rate

For $\vec{V} = (W(x_3), 0, 0)$, the rate-of-strain tensor $D = \tfrac{1}{2}(\nabla\vec{V} + \nabla\vec{V}^\top)$ has a single nonzero off-diagonal entry $D_{13} = \tfrac{1}{2}W'$, giving the scalar shear rate:

$$\dot\gamma = \sqrt{2\,D:D} = |W'|$$

### Generalized Newtonian Governing Equation

With the generalized Newtonian constitutive law $\tau_{ij} = \eta(\dot\gamma)\left(\partial_i V_j + \partial_j V_i\right)$, the momentum equation reduces to:

$$\boxed{\left(\eta(\dot\gamma)\, W'\right)' = -A}$$

where primes denote $d/dx_3$. This is the starting point for all non-Newtonian plane Poiseuille derivations.

---

## 2. General Solution for Arbitrary $\eta(\dot\gamma)$

> **This section derives a universal formula valid for any generalized Newtonian fluid.** The specific power-law, CY, and Casson solutions in §3–§5 are obtained by substituting their respective $\eta(\dot\gamma)$.

### First Integration

By symmetry, $W$ is even and $W'(0) = 0$ (maximum at centerline). For $x_3 \in [-R, 0]$: $W' \geq 0$, so $\dot\gamma = W' \equiv u \geq 0$.

Integrate the governing equation once:

$$\eta(u)\cdot u = -A\,x_3 + C_1$$

The symmetry condition $W'(0) = 0 \Rightarrow u(0) = 0 \Rightarrow C_1 = 0$:

$$\boxed{f(u) = A\,|x_3|}, \qquad f(u) \equiv \eta(u)\cdot u$$

This defines $u = u(s)$ **implicitly** as a function of $s = |x_3| \geq 0$:

$$u(s) = f^{-1}(A\,s)$$

### Second Integration

Using $W(-R) = 0$ and $W'(x_3) = u(|x_3|)$ for $x_3 \in [-R,0]$:

$$W(x_3) = \int_{|x_3|}^{R} u(s)\,ds$$

Substituting $\xi = A\,s$ (so $ds = d\xi/A$) and then $\xi = f(u)$ (so $d\xi = f'(u)\,du$):

$$W(x_3) = \frac{1}{A}\int_{A|x_3|}^{AR} f^{-1}(\xi)\,d\xi = \frac{1}{A}\int_{u_0}^{u_R} u\,f'(u)\,du$$

where $u_0 = f^{-1}(A|x_3|)$ and $u_R = f^{-1}(AR)$. Integration by parts ($\int u\,f'\,du = uf - \int f\,du$):

$$\boxed{W(x_3) = R\,u_R - |x_3|\,u_0 + \frac{1}{A}\Big[F(u_0) - F(u_R)\Big]}$$

where:

| Symbol | Definition |
|--------|-----------|
| $f(u) = \eta(u)\cdot u$ | constitutive flux function |
| $F(u) = \displaystyle\int_0^u \eta(t)\cdot t\,dt$ | constitutive potential |
| $u_0 = f^{-1}(A\|x_3\|)$ | shear rate at position $x_3$ |
| $u_R = f^{-1}(AR)$ | wall shear rate |

**The entire problem reduces to computing $f^{-1}$ (inversion) and $F$ (one quadrature) for each model.**

### Verification: Newtonian

**Newtonian** ($\eta = \mu$): $f(u) = \mu u$, $F(u) = \tfrac{\mu}{2}u^2$, $u_s = As/\mu$.

$$W = R\cdot\frac{AR}{\mu} - |x_3|\cdot\frac{A|x_3|}{\mu} + \frac{1}{A}\left[\frac{\mu}{2}\frac{A^2 x_3^2}{\mu^2} - \frac{\mu}{2}\frac{A^2 R^2}{\mu^2}\right] = \frac{A}{2\mu}(R^2 - x_3^2) \quad\checkmark$$

---

## 3. Power-Law Model

### Constitutive Relation

$$\eta(\dot\gamma) = K\,\dot\gamma^{\,n-1}$$

| Parameter | Code variable | Physical meaning | Units |
|-----------|--------------|-----------------|-------|
| $K$ | `K_phys` (physical), `K_lbm` (lattice) | consistency index | Pa·s$^n$ |
| $n$ | `lbm_n` | power-law index ($n<1$: shear-thinning, $n>1$: shear-thickening, $n=1$: Newtonian) | — |

### Flux Function

$$f(u) = \eta(u)\cdot u = K\,u^n$$

**Inversion:** $f(u) = As$ gives $u = (As/K)^{1/n}$ — **fully explicit**, no root-finding needed.

### Constitutive Potential $F(u)$

$$F(u) = \int_0^u \eta(t)\cdot t\,dt = K\int_0^u t^n\,dt = \frac{K}{n+1}\,u^{n+1}$$

### Exact Solution

Substituting $u_0 = (A|x_3|/K)^{1/n}$, $u_R = (AR/K)^{1/n}$, and $F(u) = \tfrac{K}{n+1}u^{n+1}$ into the general formula (§2):

$$W(x_3) = R\,u_R - |x_3|\,u_0 + \frac{1}{A}\left[\frac{K}{n+1}u_0^{n+1} - \frac{K}{n+1}u_R^{n+1}\right]$$

After simplification (noting $u_0^{n+1} = (A|x_3|/K)^{(n+1)/n}$ and factoring $(A/K)^{1/n}$):

$$\boxed{W(x_3) = \left(\frac{A}{K}\right)^{1/n}\!\frac{n}{n+1}\left(R^{(n+1)/n} - |x_3|^{(n+1)/n}\right)}$$

> **Nature of the solution:** The power-law model yields a **fully explicit exact solution** — no inversion, no quadrature, no special functions. The velocity profile is a generalized parabola with exponent $(n+1)/n > 1$.

---

## 4. Carreau–Yasuda (CY) Model

### Constitutive Relation

As implemented in `include/lbm3d/nonNewtonian.h` (line 500):

$$\eta(\dot\gamma) = \eta_\infty + (\eta_0 - \eta_\infty)\left[1 + (\lambda\,\dot\gamma)^a\right]^{\frac{n-1}{a}}$$

| Parameter | Code variable | Physical meaning | Units |
|-----------|--------------|-----------------|-------|
| $\eta_\infty$ | `eta_inf_phys` (physical), `lbm_nu_inf` (lattice) | infinite-shear (high-shear) viscosity | Pa·s |
| $\eta_0$ | `eta_0_phys` (physical), `lbmViscosity` (lattice) | zero-shear viscosity | Pa·s |
| $\lambda$ | `lambda_phys` (physical), `lbm_lambda` (lattice) | relaxation time constant | s |
| $a$ | `lbm_a` | Yasuda (transition) parameter | — |
| $n$ | `lbm_n` | power-law index ($n<1$: shear-thinning) | — |

### Flux Function

$$f(u) = \left[\eta_\infty + (\eta_0 - \eta_\infty)\left(1 + (\lambda u)^a\right)^{\frac{n-1}{a}}\right]u$$

**Inversion:** $f(u) = As$ is transcendental for general CY parameters. The solution $u(s)$ is defined implicitly as the unique positive root (monotonicity of $f$ for $u > 0$ guarantees uniqueness). No closed-form $f^{-1}$ exists in general.

### Constitutive Potential $F(u)$

$$F(u) = \frac{\eta_\infty\, u^2}{2} + (\eta_0 - \eta_\infty)\int_0^u t\left(1 + (\lambda t)^a\right)^{\frac{n-1}{a}}dt$$

**Evaluating the integral** via $w = (\lambda t)^a$, $t = w^{1/a}/\lambda$, $dt = \frac{1}{a\lambda}w^{1/a-1}dw$:

$$\int_0^u t\left(1 + (\lambda t)^a\right)^{\frac{n-1}{a}}dt = \frac{1}{a\lambda^2}\int_0^{(\lambda u)^a} w^{\frac{2}{a}-1}(1+w)^{\frac{n-1}{a}}\,dw$$

Using the identity $\displaystyle\int_0^X w^{p-1}(1+w)^q\,dw = \frac{X^p}{p}\;{}_2F_1\!\big(p,\,-q;\;p+1;\;-X\big)$ with $p = 2/a$, $q = (n-1)/a$:

$$\boxed{F(u) = \frac{u^2}{2}\left[\eta_\infty + (\eta_0 - \eta_\infty)\;{}_2F_1\!\left(\frac{2}{a},\,\frac{1-n}{a};\;1+\frac{2}{a};\;-(\lambda u)^a\right)\right]}$$

where ${}_2F_1$ is the Gauss hypergeometric function.

### Exact Solution

$$\boxed{W(x_3) = R\,u_R - |x_3|\,u_0 + \frac{1}{2A}\Big[u_0^2\,\Phi(u_0) - u_R^2\,\Phi(u_R)\Big]}$$

where:

$$\Phi(u) = \eta_\infty + (\eta_0 - \eta_\infty)\;{}_2F_1\!\left(\frac{2}{a},\,\frac{1-n}{a};\;1+\frac{2}{a};\;-(\lambda u)^a\right)$$

and $u_0$, $u_R$ are the unique positive solutions of:

$$\left[\eta_\infty + (\eta_0 - \eta_\infty)\left(1 + (\lambda u)^a\right)^{\frac{n-1}{a}}\right]u = A|x_3| \quad\text{and}\quad = AR$$

> **Nature of the solution:** The CY model yields an **exact but implicit** solution. The velocity profile $W(x_3)$ is expressed in closed form once $u_0, u_R$ are determined (by a monotone one-dimensional root solve at each evaluation point). The constitutive potential $F(u)$ is explicit in terms of ${}_2F_1$.

### Special Case: Carreau Model ($a = 2$)

When $a = 2$ (the original Carreau model):

$$\Phi(u) = \eta_\infty + (\eta_0 - \eta_\infty)\;{}_2F_1\!\left(1,\,\frac{1-n}{2};\;2;\;-(\lambda u)^2\right)$$

---

## 5. Casson Model

### Constitutive Relation

As implemented in `include/lbm3d/nonNewtonian.h` (lines 503–509):

$$\eta(\dot\gamma) = \frac{\left(k_0 + k_1\sqrt{\dot\gamma}\right)^2}{\dot\gamma}$$

This is the **standard Casson model** with yield stress $\tau_y = k_0^2$ and plastic viscosity $\eta_C = k_1^2$. The constitutive relation can be rewritten as $\eta = (\sqrt{\tau_y} + \sqrt{\eta_C\dot\gamma})^2/\dot\gamma$.

| Parameter | Code variable | Physical meaning | Units |
|-----------|--------------|-----------------|-------|
| $k_0$ | `tau_y_phys` → $k_0 = \sqrt{\tau_y/\rho}\cdot \delta_t/\delta_x$ (lattice), `lbm_k0` | $\sqrt{\tau_y/\rho}\cdot \delta_t/\delta_x$ — yield-stress square root in lattice units | (lattice) |
| $k_1$ | `lbmViscosity` → $k_1 = \sqrt{\nu_\text{lbm}}$ (lattice), `lbm_k1` | $\sqrt{\eta_C/\rho}$ square root (plastic-viscosity) | (lattice) |
| $\tau_y$ | `tau_y_phys` (physical) | yield stress | Pa |
| $\eta_C$ | `eta_C_phys` (physical) | plastic viscosity | Pa·s |

**Yield stress and plug flow.** At low shear rates, $\eta \to \infty$ as $\dot\gamma \to 0$, but the shear stress $\tau = \eta\dot\gamma = (k_0 + k_1\sqrt{\dot\gamma})^2 \to k_0^2 = \tau_y > 0$. For positions where the imposed shear stress $A|x_3|$ is below yield ($|x_3| < k_0^2/A$), there is no flow — a **plug region** with $u = 0$ and constant velocity $W = W_\text{max}$.

### Flux Function

$$f(u) = \eta(u)\cdot u = \frac{(k_0 + k_1\sqrt{u})^2}{u}\cdot u = (k_0 + k_1\sqrt{u})^2$$

This is a **perfect square** — a key simplification that makes the inversion trivial.

### Constitutive Potential $F(u)$

$$F(u) = \int_0^u f(t)\,dt = \int_0^u (k_0 + k_1\sqrt{t})^2\,dt = \int_0^u \left(k_0^2 + 2k_0 k_1\sqrt{t} + k_1^2 t\right)dt$$

$$\boxed{F(u) = k_0^2\,u + \frac{4}{3}\,k_0 k_1\,u^{3/2} + \frac{1}{2}\,k_1^2\,u^2}$$

**Verification:** $F'(u) = k_0^2 + 2k_0 k_1\sqrt{u} + k_1^2 u = (k_0 + k_1\sqrt{u})^2 = f(u)$ ✓

### Inversion

Setting $f(u) = As$ and solving:

$$(k_0 + k_1\sqrt{u})^2 = As \quad\Longrightarrow\quad \sqrt{u} = \frac{\sqrt{As} - k_0}{k_1}$$

$$\boxed{u(s) = \left(\frac{\sqrt{As} - k_0}{k_1}\right)^2}$$

This requires $As \geq k_0^2 = \tau_y$ (yield condition). For $As < k_0^2$ (i.e., $|x_3| < k_0^2/A$), the **plug region**, $u = 0$.

### Exact Solution

For $|x_3| \geq k_0^2/A$ (yield region):

$$\boxed{W(x_3) = R\,u_R - |x_3|\,u_0 + \frac{1}{A}\Big[F(u_0) - F(u_R)\Big]}$$

where:
- $u_0 = \left(\dfrac{\sqrt{A|x_3|} - k_0}{k_1}\right)^2$, $\quad u_R = \left(\dfrac{\sqrt{AR} - k_0}{k_1}\right)^2$
- $F(u) = k_0^2\,u + \tfrac{4}{3}\,k_0 k_1\,u^{3/2} + \tfrac{1}{2}\,k_1^2\,u^2$

For $|x_3| < k_0^2/A$ (plug region), $u_0 = 0$:

$$W(x_3) = W_\text{max} = R\,u_R - \frac{F(u_R)}{A}$$

> **Nature of the solution:** The standard Casson model yields a **fully explicit exact solution** — the inversion is a simple square root (no cubic, no Cardano), and $F(u)$ is an elementary polynomial. The solution has a plug-flow core of half-width $k_0^2/A$.

---

## 6. Pipe (Circular Cross-Section) Solutions

For axisymmetric pipe flow $W = W(r)$, $r \in [0, R]$, the governing equation becomes:

$$\frac{1}{r}\frac{d}{dr}\!\left(r\,\eta\,W'\right) = -A$$

Integrating with regularity $W'(0) = 0$:

$$\eta(u)\cdot u = \frac{A\,r}{2}$$

This is the **same** as the plane Poiseuille equation with $A \to A/2$ and $|x_3| \to r$.

### General Pipe Solution

$$\boxed{W(r) = R\,\tilde u_R - r\,\tilde u_r + \frac{2}{A}\Big[F(\tilde u_r) - F(\tilde u_R)\Big]}$$

where $\tilde u_r = f^{-1}(Ar/2)$, $\tilde u_R = f^{-1}(AR/2)$.

### CY Pipe Solution

Same formula with $\tilde u_r, \tilde u_R$ from $\left[\eta_\infty + (\eta_0-\eta_\infty)(1+(\lambda u)^a)^{(n-1)/a}\right]u = Ar/2$ (resp. $AR/2$).

### Casson Pipe Solution

Same formula with $u_r = \left(\dfrac{\sqrt{Ar/2} - k_0}{k_1}\right)^2$, $u_R = \left(\dfrac{\sqrt{AR/2} - k_0}{k_1}\right)^2$ (valid for $Ar/2 \geq k_0^2$; plug region $r < 2k_0^2/A$).

### Newtonian Verification (Pipe)

$W(r) = \frac{A}{4\eta_0}(R^2 - r^2)$ — the classical Hagen–Poiseuille parabola ✓

---

## 7. Volumetric Flow Rate

### Plane Poiseuille

$$Q = 2\int_0^R t\,u(t)\,dt = \frac{2}{A^2}\int_0^{AR}\xi\,f^{-1}(\xi)\,d\xi$$

After substitution $\xi = f(u)$ and integration by parts:

$$\boxed{Q = R^2\,u_R - \frac{1}{A^2}\int_0^{u_R} f(u)^2\,du}$$

### Newtonian Check

$f(u) = \mu u$, $u_R = AR/\mu$:

$$Q = R^2\cdot\frac{AR}{\mu} - \frac{1}{A^2}\cdot\frac{\mu^2}{3}\left(\frac{AR}{\mu}\right)^3 = \frac{AR^3}{\mu} - \frac{AR^3}{3\mu} = \frac{2AR^3}{3\mu} \quad\checkmark$$

### Power-Law

$$Q = \frac{2n}{2n+1}\,R^2\left(\frac{AR}{K}\right)^{1/n}$$

### CY and Casson

$Q$ requires evaluating $\int_0^{u_R} f(u)^2\,du$. For **Casson**, $f(u) = (k_0 + k_1\sqrt{u})^2$ is a perfect square, so $f(u)^2 = (k_0 + k_1\sqrt{u})^4$ is a polynomial in $v = \sqrt{u}$, integrable in closed form. For **CY**, it involves products of hypergeometric functions and is best left in integral form or evaluated numerically.

---

## 8. Limiting Cases and Verification

### Power-Law Model

| Limit | Behavior | Verification |
|-------|----------|-------------|
| $n = 1$ | $\eta = K$ (Newtonian, $\mu = K$) | $W = \frac{A}{2K}(R^2 - x_3^2)$ ✓ (matches §2) |
| $n \to 0^+$ | $\eta \to K/\dot\gamma$ (τ → K constant) | Singular limit: flat plug for $\|x_3\| < K/A$, sharp drop near wall; no simple closed form |
| $n \to \infty$ | $\eta \to \infty$ for $\dot\gamma > 1$, $\eta \to 0$ for $\dot\gamma < 1$ | Rigid-plastic, plug flow |
| $n < 1$ | Shear-thinning (pseudoplastic) | Flatter profile than parabola, $W \propto (R^{(n+1)/n} - \lvert x_3\rvert^{(n+1)/n})$ |
| $n > 1$ | Shear-thickening (dilatant) | Sharper profile than parabola |

### CY Model

| Limit | Behavior | Verification |
|-------|----------|-------------|
| $n \to 1$ | $\eta \to \eta_0$ (Newtonian) | ${}_2F_1(\cdot, 0; \cdot; \cdot) = 1$, $F = \eta_0 u^2/2$, $W = \frac{A}{2\eta_0}(R^2 - x_3^2)$ ✓ |
| $\lambda \to 0$ | $\eta \to \eta_0$ (Newtonian) | $(1+0)^{(\cdot)} = 1$ ✓ |
| $\dot\gamma \to \infty$, $n < 1$ | $\eta \to \eta_\infty$ | $(1+(\lambda u)^a)^{(n-1)/a} \approx (\lambda u)^{n-1} \to 0$ ✓ |
| Power-law regime | $\eta \approx (\eta_0 - \eta_\infty)\lambda^{n-1} u^{n-1}$ | Intermediate $\dot\gamma$ where $(\lambda u)^a \gg 1$ |

### Casson Model

| Limit | Behavior | Verification |
|-------|----------|-------------|
| $k_0 \to 0$ | $\eta = k_1^2$ (Newtonian, $\mu = k_1^2$) | $f = k_1^2 u$, $F = \tfrac{1}{2}k_1^2 u^2$, $W = \frac{A}{2k_1^2}(R^2 - x_3^2)$ ✓ |
| $k_1 \to 0$ | $\eta = k_0^2/u$ (pure yield stress) | $f = k_0^2$ (const), no flow for $AR < k_0^2$ |

**Newtonian limit** ($k_0 = 0$): $f(u) = k_1^2 u$, $F(u) = \tfrac{1}{2}k_1^2 u^2$, $u_s = As/k_1^2$.

$$W = \frac{A}{2k_1^2}(R^2 - x_3^2) \quad\checkmark$$

This matches the Newtonian formula with $\mu = k_1^2$, confirming consistency with §2.

---

## 9. Relationship to the TNL-LBM Code

### Code Location

| Component | File | Line(s) |
|-----------|------|---------|
| `copyQuantities` (SD→KS transfer) | `include/lbm3d/nonNewtonian.h` | 450–456 |
| `compute_dnu` lambda (forcing kernel) | `include/lbm3d/nonNewtonian.h` | 488–511 |
| Power-law formula | `include/lbm3d/nonNewtonian.h` | 493–499 |
| CY formula | `include/lbm3d/nonNewtonian.h` | 500–502 |
| Casson formula | `include/lbm3d/nonNewtonian.h` | 503–509 |
| Forcing divergence (product rule) | `include/lbm3d/nonNewtonian.h` | 513–597 |
| `poiseuille_W` (general analytical solution) | `sim_NSE/sim_nonnewtonian_poiseuille.cu` | 69–75 |
| `CY_Constitutive` struct | `sim_NSE/sim_nonnewtonian_poiseuille.cu` | 94–120 |
| `Casson_Constitutive` struct | `sim_NSE/sim_nonnewtonian_poiseuille.cu` | 122–150 |
| Power-law analytical solution | `sim_NSE/sim_nonnewtonian_poiseuille.cu` | 256–262 |
| CY analytical solution | `sim_NSE/sim_nonnewtonian_poiseuille.cu` | 264–286 |
| Casson analytical solution | `sim_NSE/sim_nonnewtonian_poiseuille.cu` | 288–307 |
| Unit conversions and non-dimensionalization | `sim_NSE/sim_nonnewtonian_poiseuille.cu` | 630–685 (PL), 686–768 (CY), 769–830 (Casson) |

### Parameter Mapping

The code computes the physical shear rate `gamma = sqrt(2 * D:D)` from the strain-rate tensor components. For the plane Poiseuille benchmark ($\vec{V} = (W(x_3),0,0)$), this reduces to `gamma = |W'|` exactly, matching the physical shear rate $\dot\gamma$ used in the analytical derivation. The model parameters live only in the data struct `SD` (not the kernel struct `KS`) and are used in their standard physical forms without any rescaling.

| Physical parameter | Units | CY code variable | Casson code variable |
|---|---|---|---|
| $\eta_\infty$ (infinite-shear viscosity) | Pa·s | `lbm_nu_inf` | — |
| $\eta_0$ (zero-shear viscosity) | Pa·s | `lbmViscosity` | — |
| $\lambda$ (relaxation time) | s | `lbm_lambda` | — |
| $a$ (Yasuda parameter) | — | `lbm_a` | — |
| $n$ (power-law index) | — | `lbm_n` | — |
| $\tau_y$ (yield stress, $= k_0^2 \rho / (\delta_t/\delta_x)^2$) | Pa | — | `lbm_k0` |
| $\eta_C$ (plastic viscosity, $= k_1^2 \rho \cdot \delta_x^2/\delta_t$) | Pa·s | — | `lbm_k1` |

### Non-Dimensionalization (Physical → Lattice Units)

The simulation converts physical parameters to lattice units via the lattice spacing $\delta_x$ and time step $\delta_t$. Each model defines $\delta_t$ so that the collision reference viscosity equals `lbmViscosity` ($\nu_\text{lbm}$). All three models use **diffusive scaling** ($\delta_t \propto \delta_x^2$):

**Power-law:** $K_\text{phys}$ has units Pa·s$^n$, so $K_\text{phys}/\rho$ is not a kinematic viscosity for $n \neq 1$. A reference shear rate $\dot\gamma_\text{ref} = u_{\max}/R$ converts it to a proper kinematic viscosity:

$$\nu_\text{ref} = \frac{K_\text{phys}}{\rho} \cdot \dot\gamma_\text{ref}^{n-1} \quad [\text{m}^2/\text{s}], \qquad \delta_t = \frac{\nu_\text{lbm} \cdot \delta_x^2}{\nu_\text{ref}}$$

$$K_\text{lbm} = \frac{K_\text{phys}}{\rho} \cdot \frac{\delta_t^{2-n}}{\delta_x^2} \quad (\text{varies with resolution for } n \neq 1), \qquad \nu_\text{lbm} = K_\text{lbm} \cdot \dot\gamma_{\text{ref},\text{lbm}}^{n-1} \quad (\text{invariant})$$

For $n=1$: $\nu_\text{ref} = K_\text{phys}/\rho$, $K_\text{lbm} = \nu_\text{lbm}$ (identical to Newtonian). For $n \neq 1$: $K_\text{lbm}$ varies with resolution, but $\nu_\text{lbm}$ (the viscosity at $\dot\gamma_\text{ref}$, which controls $\omega$) stays invariant. The generalized Reynolds number $Re_\text{ref} = 2 \rho \, R^n \cdot u_{\max}^{2-n} / K_\text{phys}$ is resolution-invariant.

**Carreau–Yasuda:** $\delta_t = \frac{\nu_\text{lbm} \cdot \delta_x^2}{\eta_0/\rho}$, so:

$$\nu_{0,\text{lbm}} = \nu_\text{lbm}, \qquad \nu_{\infty,\text{lbm}} = \frac{\eta_\infty}{\rho}\cdot\frac{\delta_t}{\delta_x^2}, \qquad \lambda_\text{lbm} = \frac{\lambda_\text{phys}}{\delta_t}$$

**Casson:** $\delta_t = \frac{\nu_\text{lbm} \cdot \delta_x^2}{\eta_C/\rho}$, so:

$$k_{0,\text{lbm}} = \sqrt{\frac{\tau_y}{\rho}}\cdot\frac{\delta_t}{\delta_x}, \qquad k_{1,\text{lbm}} = \sqrt{\nu_\text{lbm}}$$

### Diffusive Scaling

The simulation uses **diffusive scaling** ($\delta_t \propto \delta_x^2$) for all three models, keeping $\nu_\text{lbm}$ constant across resolutions. This is the standard scaling for incompressible LBM (as opposed to acoustic scaling $\delta_t \propto \delta_x$, which keeps the Mach number constant):

$$\nu_\text{ref} = \nu_\text{lbm} \cdot \frac{\delta_x^2}{\delta_t} = \text{const} \quad\Longrightarrow\quad \delta_t \propto \delta_x^2$$

For CY and Casson, $\nu_\text{ref}$ is a material parameter ($\eta_0/\rho$ or $\eta_C/\rho$). For Power-law, $\nu_\text{ref} = (K_\text{phys}/\rho) \cdot (u_{\max}/R)^{n-1}$ depends on the flow (but not on $\delta_x$), so $Re_\text{ref}$ is resolution-invariant for all models.

Consequences for fixed physical quantities when refining the grid ($\delta_x \to \delta_x/2$):

| Quantity | Scaling | When $u_{\max,\text{phys}}$ is fixed |
|---|---|---|
| $\delta_t$ | $\propto \delta_x^2$ | decreases by $4\times$ |
| $u_{\max,\text{lbm}} = u_{\max,\text{phys}}\cdot \delta_t/\delta_x$ | $\propto \delta_x$ | decreases by $2\times$ |
| $A_\text{lbm} = (A_\text{phys}/\rho)\cdot \delta_t^2/\delta_x$ | $\propto \delta_x^3$ | decreases by $8\times$ |
| $\dot\gamma_\text{lbm} = \dot\gamma_\text{phys}\cdot \delta_t$ | $\propto \delta_x^2$ | decreases by $4\times$ |
| $\omega = 1/(3\nu_\text{lbm}+0.5)$ | const | unchanged |

### Numerical Regularization

The Casson viscosity $\eta = (k_0 + k_1\sqrt{\dot\gamma})^2/\dot\gamma$ diverges as $\dot\gamma \to 0$ (the yield-stress singularity). The LBM kernel regularizes this by clamping the shear rate to a floor $\dot\gamma_\text{cap} = \max(\dot\gamma, \epsilon)$ with $\epsilon = 10^{-10}$, then evaluating the standard Casson formula at the clamped value:

$$\eta_\text{reg}(\dot\gamma) = \frac{(k_0 + k_1\sqrt{\dot\gamma_\text{cap}})^2}{\dot\gamma_\text{cap}}, \qquad \dot\gamma_\text{cap} = \max(\dot\gamma, \, 10^{-10})$$

**Properties:**
- $\eta_\text{reg}(0) = (k_0 + k_1\sqrt{\epsilon})^2/\epsilon \approx k_0^2/\epsilon$ — bounded (for $k_0 = 0.01$, $\epsilon = 10^{-10}$: $\eta_\text{reg}(0) \approx 10^6$, large but finite).
- $\eta_\text{reg}(\dot\gamma) = \eta(\dot\gamma)$ for $\dot\gamma \geq 10^{-10}$ — exact standard Casson behavior everywhere except a sub-FP32-noise region at the plug.
- Single smooth formula (no branch); the $\max$ introduces a kink only at $\dot\gamma = 10^{-10}$, far below FP32 noise level, so the product-rule finite differences are unaffected in practice.

### Analytical Solution and Driving-Force Computation

The analytical solutions operate entirely in lattice units: $u_\text{lbm} = u_\text{phys}\cdot \delta_t/\delta_x$, $A_\text{lbm} = (A_\text{phys}/\rho)\cdot \delta_t^2/\delta_x$, and the lattice shear rate $u = |W'|_\text{lbm}$. The driving force $A_\text{lbm}$ is computed from either the Reynolds number ($u_{\max,\text{lbm}} = Re\cdot\nu_\text{lbm}/(2R_\text{lbm})$) or the physical $u_{\max}$ ($u_{\max,\text{lbm}} = u_{\max,\text{phys}}\cdot \delta_t/\delta_x$).

**Power-law** has a closed-form inverse: $A_\text{lbm} = K_\text{lbm} \cdot \left(\frac{(n+1) \cdot u_{\max,\text{lbm}}}{n \cdot R_\text{lbm}^{(n+1)/n}}\right)^n$.

**CY and Casson** have no closed-form $A(u_\text{max})$ because their viscosity $\eta(u)$ is a nonlinear function of the shear rate. The code uses bisection root-finding on the analytical Poiseuille solution $W(0; A) = u_{\max,\text{lbm}}$:

$$W(s; A) = R \cdot u_R - s \cdot u_0 + \frac{F(u_0) - F(u_R)}{A}$$

where $u_0 = f^{-1}(A \cdot s)$ and $u_R = f^{-1}(A \cdot R)$ are found by the monotone solver `solve_monotone` (bisection on the monotone flux function $f(u) = \eta(u)\cdot u$), and $F(u) = \int_0^u \eta(t)\cdot t \, dt$ is the constitutive potential. The solver handles both the standard case ($f(0) = 0$) and the yield-stress case ($f(0) > 0$, Casson plug region) by returning $u = 0$ when $f(0) \geq \text{rhs}$.

**CY constitutive law**: $\eta(u) = \eta_\infty + (\eta_0 - \eta_\infty) \cdot (1 + (\lambda u)^a)^{(n-1)/a}$. The potential $F(u)$ has no elementary closed form for general $a$, $n$, so it is evaluated numerically by composite Simpson's rule (500 subintervals). The bisection on $A$ starts with $A_\text{hi} = 1$ and doubles until $W(0; A_\text{hi}) \geq u_\text{max}$, then runs 100 bisection iterations — sufficient for double-precision convergence.

**Casson constitutive law**: $\eta(u) = (k_0 + k_1\sqrt{u})^2/u$. Here the potential $F(u) = k_0^2 u + \frac{4}{3} k_0 k_1 u^{3/2} + \frac{1}{2} k_1^2 u^2$ is analytical (no Simpson integration needed), but $A(u_\text{max})$ still has no closed form, so the same bisection loop is used.

---

## Appendix: Hypergeometric Identity

**Identity used in §4:**

$$\int_0^X w^{p-1}(1+w)^q\,dw = \frac{X^p}{p}\;{}_2F_1\!\left(p,\,-q;\;p+1;\;-X\right)$$

**Proof sketch:** Expand $(1+w)^q = \sum_{k=0}^\infty \frac{(-q)_k}{k!}(-w)^k$ (binomial series), integrate termwise:

$$\int_0^X w^{p-1}\sum_{k=0}^\infty \frac{(-q)_k}{k!}(-1)^k w^k\,dw = \sum_{k=0}^\infty \frac{(-q)_k(-1)^k}{k!(p+k)} X^{p+k}$$

Using $\frac{p}{p+k} = \frac{(p)_k}{(p+1)_k}$ (Pochhammer ratio) and combining $(-1)^k X^k = (-X)^k$:

$$= \frac{X^p}{p}\sum_{k=0}^\infty \frac{(p)_k\,(-q)_k}{(p+1)_k\,k!}(-X)^k = \frac{X^p}{p}\;{}_2F_1(p,-q;\,p+1;\,-X) \quad\square$$

---

## References

- **Power-Law (Ostwald–de Waele):** [RheoJAX model handbook](https://rheojax.readthedocs.io/en/latest/models/flow/power_law.html) — constitutive equation, parameter table, pipe-flow velocity profile.
- **Carreau–Yasuda model:** [RheoJAX model handbook](https://rheojax.readthedocs.io/en/latest/models/flow/carreau_yasuda.html) — constitutive equation, limiting cases, relation to Carreau (a=2).
