"""Interactive Streamlit app for non-Newtonian Poiseuille flow.

Displays viscosity vs shear rate and velocity profiles for power-law,
Carreau-Yasuda (CY), and Casson fluid models, with dimensionless quantity
verification.

Run: streamlit run streamlit_app.py
"""

from __future__ import annotations

import json
import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Final

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import scipy.optimize
import streamlit as st

# --------------------------------------------------------------------------- #
# Constants (from sim_nonnewtonian_poiseuille.cu)
# --------------------------------------------------------------------------- #

BLOCK_SIZE: Final[int] = 32
C_S: Final[float] = 1.0 / math.sqrt(3.0)
OMEGA_MIN: Final[float] = 0.5
OMEGA_MAX: Final[float] = 1.9
MA_MAX: Final[float] = 0.1
RHO: Final[float] = 1000.0

DEFAULTS: Final[dict[str, float]] = {
    "lbm_viscosity": 0.05,
    "n_pl": 1.0,
    "K_phys": 0.01,
    "eta_inf": 0.01,
    "eta_0": 0.1,
    "lambda_cy": 1.0,
    "a_cy": 2.0,
    "n_cy": 0.5,
    "tau_y": 0.01,
    "eta_C": 0.01,
}

MODEL_KEYS: Final[dict[str, list[str]]] = {
    "pl": ["n_pl", "K_phys"],
    "cy": ["eta_inf", "eta_0", "lambda_cy", "a_cy", "n_cy"],
    "cas": ["tau_y", "eta_C"],
}


# --------------------------------------------------------------------------- #
# Bisection solver (from sim_nonnewtonian_poiseuille.cu: solve_monotone)
# --------------------------------------------------------------------------- #


def solve_monotone(
    f: Callable[[float], float], rhs: float, tol: float = 1e-12
) -> float:
    """Solve f(u) = rhs for u >= 0, f monotone increasing. Returns 0 if rhs <= 0
    or if f(0) >= rhs (plug region for yield-stress fluids)."""
    if rhs <= 0.0:
        return 0.0
    if f(0.0) >= rhs:
        return 0.0
    lo = 0.0
    hi = max(rhs, 1.0)
    while f(hi) < rhs:
        hi *= 2.0
    for _ in range(200):
        mid = lo + (hi - lo) * 0.5
        if f(mid) < rhs:
            lo = mid
        else:
            hi = mid
        if hi - lo < tol:
            break
    return lo + (hi - lo) * 0.5


def simpson_integral(g: Callable[[float], float], u: float, n: int = 500) -> float:
    """Simpson's rule for integral_0^u g(t) dt (from sim source)."""
    if u <= 0.0:
        return 0.0
    h = u / n
    s = g(0.0) + g(u)
    for i in range(1, n):
        t = i * h
        s += (2.0 if i % 2 == 0 else 4.0) * g(t)
    return s * h / 3.0


# --------------------------------------------------------------------------- #
# Constitutive models
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class PowerLaw:
    """eta(gamma) = K * gamma^(n-1)."""

    K: float
    n: float

    def nu(self, gamma: float) -> float:
        if gamma <= 0.0:
            return math.inf if self.n < 1.0 else (self.K if self.n == 1.0 else 0.0)
        return self.K * gamma ** (self.n - 1.0)

    def flux(self, u: float) -> float:
        return self.K * u**self.n

    def pot(self, u: float) -> float:
        return self.K * u ** (self.n + 1.0) / (self.n + 1.0)


@dataclass(frozen=True, slots=True)
class CarreauYasuda:
    """nu(gamma) = nu_inf + (nu_0 - nu_inf) * (1 + (lambda*gamma)^a)^((n-1)/a)."""

    nu_inf: float
    nu_0: float
    lambda_: float
    a: float
    n: float

    def nu(self, gamma: float) -> float:
        if gamma <= 0.0:
            return self.nu_0
        return self.nu_inf + (self.nu_0 - self.nu_inf) * (
            1.0 + (self.lambda_ * gamma) ** self.a
        ) ** ((self.n - 1.0) / self.a)

    def flux(self, u: float) -> float:
        return self.nu(u) * u

    def pot(self, u: float) -> float:
        return simpson_integral(lambda t: self.nu(t) * t, u)


@dataclass(frozen=True, slots=True)
class Casson:
    """eta(gamma) = (k0 + k1*sqrt(gamma))^2 / gamma."""

    k0: float
    k1: float

    def nu(self, gamma: float) -> float:
        if gamma <= 0.0:
            return math.inf
        sg = math.sqrt(gamma)
        return (self.k0 + self.k1 * sg) ** 2 / gamma

    def flux(self, u: float) -> float:
        # No guard at u=0: flux(0) = k0^2 (yield-stress intercept).
        # solve_monotone uses f(0) >= rhs to detect the plug region.
        # Matches sim_nonnewtonian_poiseuille.cu:135-138.
        su = math.sqrt(u) if u > 0.0 else 0.0
        return (self.k0 + self.k1 * su) ** 2

    def pot(self, u: float) -> float:
        su = math.sqrt(u)
        return (
            self.k0**2 * u
            + (4.0 / 3.0) * self.k0 * self.k1 * u * su
            + 0.5 * self.k1**2 * u * u
        )


ConstitutiveModel = PowerLaw | CarreauYasuda | Casson


# --------------------------------------------------------------------------- #
# Analytical Poiseuille velocity (from sim_nonnewtonian_poiseuille.cu)
# --------------------------------------------------------------------------- #


def poiseuille_W(
    s: float,
    R: float,
    A: float,
    flux: Callable[[float], float],
    pot: Callable[[float], float],
) -> float:
    """General non-Newtonian plane Poiseuille velocity at |z_rel| = s."""
    if abs(s) >= R:
        return 0.0
    s_abs = abs(s)
    u_0 = solve_monotone(flux, A * s_abs)
    u_R = solve_monotone(flux, A * R)
    return R * u_R - s_abs * u_0 + (pot(u_0) - pot(u_R)) / A


def invert_A_lbm_from_umax(
    u_max_lbm: float,
    R_lbm: float,
    flux_fn: Callable[[float], float],
    pot_fn: Callable[[float], float],
) -> float:
    """Find A_lbm such that poiseuille_W(0, R_lbm, A_lbm, ...) = u_max_lbm.

    Matches the sim's root-find approach (sim_nonnewtonian_poiseuille.cu:658-685).
    """

    def w_center(a: float) -> float:
        return poiseuille_W(0.0, R_lbm, a, flux_fn, pot_fn)

    a_lo = 1e-20
    a_hi = 1.0
    while w_center(a_hi) < u_max_lbm:
        a_hi *= 2.0
        if a_hi > 1e10:
            return a_hi
    for _ in range(200):
        a_mid = (a_lo + a_hi) * 0.5
        if w_center(a_mid) < u_max_lbm:
            a_lo = a_mid
        else:
            a_hi = a_mid
        if a_hi - a_lo < 1e-15 * a_mid:
            break
    return (a_lo + a_hi) * 0.5


def powerlaw_velocity(s: float, R: float, A: float, K: float, n: float) -> float:
    """Closed-form power-law velocity.

    W = (A/K)^(1/n) * n/(n+1) * (R^((n+1)/n) - |s|^((n+1)/n))
    """
    if abs(s) >= R:
        return 0.0
    exponent = (n + 1.0) / n
    coeff = (A / K) ** (1.0 / n) * n / (n + 1.0)
    return coeff * (R**exponent - abs(s) ** exponent)


def newtonian_velocity(s: float, R: float, A: float, nu: float) -> float:
    if abs(s) >= R:
        return 0.0
    return A / (2.0 * nu) * (R * R - s * s)


# --------------------------------------------------------------------------- #
# Dimensionless quantity computation
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class DimNumbers:
    res: int
    n_fluid: int
    r_lbm: float
    r_phys: float
    phys_dl: float
    phys_dt: float
    omega: float
    u_max_lbm: float
    ma: float
    u_max_phys: float
    re: float
    re_ref: float
    t_steady_lbm: float
    t_steady_phys: float
    nu_wall: float
    nu_ratio: float
    a_lbm: float
    a_phys: float


def _pl_u_max_phys(
    drive_mode: str,
    a_phys: float | None,
    u_max_input: float | None,
    re_target: float | None,
    re_ref_target: float | None,
    K_phys: float,
    n: float,
    rho: float,
    R_phys: float,
) -> float:
    """Analytical u_max_phys for Power-law (no bisection needed).

    For all driving modes, u_max can be solved in closed form because
    Re_ref = 2·(ρ/K)·u_max^(2-n)·R^n has an analytical inverse.
    """
    if drive_mode.startswith("Body"):
        assert a_phys is not None
        return (
            (a_phys * rho / K_phys) ** (1.0 / n)
            * n
            / (n + 1.0)
            * R_phys ** ((n + 1.0) / n)
        )
    elif drive_mode.startswith("Max"):
        assert u_max_input is not None
        return u_max_input
    elif drive_mode == "Re_wall":
        assert re_target is not None
        # Re_wall = Re_ref / ((n+1)/n)^(n-1)
        re_ref = re_target * ((n + 1.0) / n) ** (n - 1.0)
        return (re_ref * K_phys / (2.0 * rho * R_phys**n)) ** (1.0 / (2.0 - n))
    else:
        assert re_ref_target is not None
        return (re_ref_target * K_phys / (2.0 * rho * R_phys**n)) ** (1.0 / (2.0 - n))


def _u_max_from_re_wall(
    re_target: float,
    r_phys: float,
    r_lbm: float,
    nu_lbm: float,
    phys_dl: float,
    dt: float,
    n: float,
    nu_fn: Callable[[float], float],
    flux_fn: Callable[[float], float],
    pot_fn: Callable[[float], float],
) -> float:
    """Find u_max_phys such that Re_wall = re_target.

    Re_wall = u_max_phys * 2*r_phys / nu_wall_phys, where nu_wall_phys depends
    on u_max (through the wall shear rate). Bisect on u_max_phys.
    """

    def re_wall(u_max_phys: float) -> float:
        u_max_lbm = u_max_phys * dt / phys_dl
        a_lbm = invert_A_lbm_from_umax(u_max_lbm, r_lbm, flux_fn, pot_fn)
        gamma_wall = solve_monotone(flux_fn, a_lbm * r_lbm)
        nu_wall_lbm = nu_fn(gamma_wall) if gamma_wall > 0 else nu_lbm
        nu_wall_phys = nu_wall_lbm * phys_dl**2 / dt
        return u_max_phys * 2.0 * r_phys / nu_wall_phys

    lo, hi = 1e-15, 1e6
    while re_wall(hi) < re_target:
        hi *= 2.0
        if hi > 1e12:
            return hi
    for _ in range(200):
        mid = (lo + hi) * 0.5
        if re_wall(mid) < re_target:
            lo = mid
        else:
            hi = mid
        if hi - lo < 1e-15 * mid:
            break
    return (lo + hi) * 0.5


def compute_dimensionless(
    *,
    nu_lbm: float,
    nu_phys: float,
    n: float,
    phys_height: float,
    res: int,
    a_phys_val: float | None = None,
    u_max_phys_val: float | None = None,
    re_target: float | None = None,
    re_ref_target: float | None = None,
    nu_fn: Callable[[float], float],
    flux_fn: Callable[[float], float],
    pot_fn: Callable[[float], float],
    is_powerlaw: bool = False,
    phys_dt_fn: Callable[[float], float] | None = None,
) -> DimNumbers:
    """Compute dimensionless quantities for one resolution.

    Exactly one of ``a_phys_val``, ``u_max_phys_val``, ``re_target``,
    or ``re_ref_target`` must be set. The others are derived per resolution.
    """
    z = BLOCK_SIZE * res
    n_fluid = z - 4
    r_lbm = n_fluid / 2.0
    r_phys = phys_height / 2.0

    phys_dl = phys_height / n_fluid
    if phys_dt_fn is not None:
        dt = phys_dt_fn(phys_dl)
    else:
        dt = nu_lbm / nu_phys * phys_dl**2
    omega = 1.0 / (3.0 * nu_lbm + 0.5)

    # --- Determine A_lbm from the driving parameter ---
    # Dimensional velocity conversion: u_lbm = u_phys * dt / dl
    # (correct for all models because dt is model-specific via phys_dt_fn)
    if a_phys_val is not None:
        a_lbm = a_phys_val * dt**2 / phys_dl
    elif is_powerlaw and n > 0.0:
        if u_max_phys_val is not None:
            u_max_phys = u_max_phys_val
        elif re_ref_target is not None:
            nu_ref_phys = nu_lbm * phys_dl**2 / dt
            u_max_phys = re_ref_target * nu_ref_phys / (2.0 * r_phys)
        else:
            assert re_target is not None
            u_max_phys = _u_max_from_re_wall(
                re_target, r_phys, r_lbm, nu_lbm, phys_dl, dt, n, nu_fn, flux_fn, pot_fn
            )
        u_max_lbm_tmp = u_max_phys * dt / phys_dl
        exponent = (n + 1.0) / n
        K_lbm_res = flux_fn(1.0)
        a_lbm = K_lbm_res * (u_max_lbm_tmp * (n + 1.0) / n * r_lbm ** (-exponent)) ** n
    else:
        if u_max_phys_val is not None:
            u_max_phys = u_max_phys_val
        elif re_ref_target is not None:
            nu_ref_phys = nu_lbm * phys_dl**2 / dt
            u_max_phys = re_ref_target * nu_ref_phys / (2.0 * r_phys)
        else:
            assert re_target is not None
            u_max_phys = _u_max_from_re_wall(
                re_target, r_phys, r_lbm, nu_lbm, phys_dl, dt, n, nu_fn, flux_fn, pot_fn
            )
        u_max_lbm_tmp = u_max_phys * dt / phys_dl
        a_lbm = invert_A_lbm_from_umax(u_max_lbm_tmp, r_lbm, flux_fn, pot_fn)

    # --- Compute u_max_lbm via poiseuille_W (exact for all models) ---
    gamma_wall = solve_monotone(flux_fn, a_lbm * r_lbm)
    nu_wall = nu_fn(gamma_wall) if gamma_wall > 0 else nu_lbm

    u_max_lbm = poiseuille_W(0.0, r_lbm, a_lbm, flux_fn, pot_fn)

    ma = u_max_lbm / C_S
    u_max_phys_out = u_max_lbm * phys_dl / dt
    nu_wall_phys = nu_wall * phys_dl**2 / dt
    # Re_wall: based on actual wall viscosity (Streamlit's primary Re)
    re_out = u_max_phys_out * 2.0 * r_phys / nu_wall_phys
    # Re_ref: based on reference (collision) viscosity nu_lbm
    # (matches C++ PHYS_VISCOSITY = nu_lbm * dl²/dt)
    nu_ref_phys = nu_lbm * phys_dl**2 / dt
    re_ref_out = u_max_phys_out * 2.0 * r_phys / nu_ref_phys
    t_steady_lbm = r_lbm**2 / nu_lbm
    t_steady_phys = t_steady_lbm * dt
    nu_ratio = nu_wall / nu_lbm
    a_phys = a_lbm * phys_dl / dt**2

    return DimNumbers(
        res=res,
        n_fluid=n_fluid,
        r_lbm=r_lbm,
        r_phys=r_phys,
        phys_dl=phys_dl,
        phys_dt=dt,
        omega=omega,
        u_max_lbm=u_max_lbm,
        ma=ma,
        u_max_phys=u_max_phys_out,
        re=re_out,
        re_ref=re_ref_out,
        t_steady_lbm=t_steady_lbm,
        t_steady_phys=t_steady_phys,
        nu_wall=nu_wall,
        nu_ratio=nu_ratio,
        a_lbm=a_lbm,
        a_phys=a_phys,
    )


# --------------------------------------------------------------------------- #
# Plotting helpers
# --------------------------------------------------------------------------- #


def plot_viscosity_multi(
    models: list[tuple[str, Callable[[float], float], str]],
    gamma_min: float,
    gamma_max: float,
) -> go.Figure:
    gammas = np.geomspace(gamma_min, gamma_max, 500)

    fig = go.Figure()
    for name, eta_fn, color in models:
        etas = np.array([eta_fn(g) for g in gammas])
        fig.add_trace(
            go.Scatter(
                x=gammas,
                y=etas,
                mode="lines",
                name=name,
                line=dict(color=color, width=2),
            )
        )
    fig.update_layout(
        title="Viscosity vs Shear rate",
        xaxis_title="Shear rate γ̇ [1/s] (log scale)",
        yaxis_title="Dynamic viscosity η [Pa·s]",
        template="plotly_white",
        height=450,
        legend=dict(yanchor="bottom", y=-0.3, xanchor="center", x=0.5, orientation="h"),
    )
    fig.update_xaxes(type="log")
    return fig


def plot_velocity_multi(
    models: list[tuple[str, Callable[[float], float], str]],
    newtonian_models: list[tuple[str, Callable[[float], float], str]] | None,
    R: float,
) -> go.Figure:
    s = np.linspace(-R, R, 401)

    fig = go.Figure()
    for name, w_fn, color in models:
        w = np.array([w_fn(si) for si in s])
        fig.add_trace(
            go.Scatter(
                x=s,
                y=w,
                mode="lines",
                name=name,
                line=dict(color=color, width=2),
            )
        )
    if newtonian_models is not None:
        for name, w_newt_fn, color in newtonian_models:
            w_newt = np.array([w_newt_fn(si) for si in s])
            label = f"{name} (Newtonian)" if name else "Newtonian"
            fig.add_trace(
                go.Scatter(
                    x=s,
                    y=w_newt,
                    mode="lines",
                    name=label,
                    line=dict(color=color, width=1.5, dash="dash"),
                )
            )
    fig.update_layout(
        title="Velocity profile",
        xaxis_title="Position z [m]",
        yaxis_title="Velocity W [m/s]",
        template="plotly_white",
        height=450,
        legend=dict(yanchor="bottom", y=-0.3, xanchor="center", x=0.5, orientation="h"),
    )
    return fig


# --------------------------------------------------------------------------- #
# Cross-model parameter fitting (velocity-profile-based)
# --------------------------------------------------------------------------- #


def fit_by_velocity_profile(
    source_model: ConstitutiveModel,
    source_dt: float,
    target_builder: Callable[[np.ndarray], tuple[ConstitutiveModel, float]],
    param_bounds: list[tuple[float, float]],
    a_phys: float,
    phys_height: float,
    dl: float,
    n_points: int = 30,
) -> np.ndarray | None:
    """Fit target model parameters by matching the normalized velocity profile.

    Both source and target profiles are computed at res=1 with the same A_phys.
    Profiles are normalized by u_max (center velocity) so the fit targets the
    SHAPE — after fitting, switching to u_max mode forces the absolute velocity
    to match, which together with the matched shape gives matching wall shear
    rate and Re_wall.

    target_builder(params) -> (model, dt)
    """
    R_lbm = (BLOCK_SIZE - 4) / 2.0
    R_phys = phys_height / 2.0

    a_lbm_src = a_phys * source_dt**2 / dl
    s_phys_arr = np.linspace(0, R_phys * 0.99, n_points)
    s_lbm_arr = s_phys_arr / dl
    w_src = np.array(
        [
            poiseuille_W(s, R_lbm, a_lbm_src, source_model.flux, source_model.pot)
            for s in s_lbm_arr
        ]
    )
    u_max_src = w_src[0]
    if u_max_src <= 0:
        return None
    w_src_norm = w_src / u_max_src

    def objective(params: np.ndarray) -> float:
        try:
            m_tgt, dt_tgt = target_builder(params)
        except Exception:
            return 1e10
        a_lbm_tgt = a_phys * dt_tgt**2 / dl
        w_tgt = np.array(
            [
                poiseuille_W(s, R_lbm, a_lbm_tgt, m_tgt.flux, m_tgt.pot)
                for s in s_lbm_arr
            ]
        )
        u_max_tgt = w_tgt[0]
        if u_max_tgt <= 0:
            return 1e10
        return float(np.sum((w_tgt / u_max_tgt - w_src_norm) ** 2))

    try:
        result = scipy.optimize.differential_evolution(
            objective,
            param_bounds,
            maxiter=100,
            tol=1e-14,
            seed=42,  # pyright: ignore[reportCallIssue]
            polish=True,
        )
        return result.x
    except Exception:
        return None


# --------------------------------------------------------------------------- #
# Per-model state computation
# --------------------------------------------------------------------------- #

COLOR_POWERLAW = "#1f77b4"
COLOR_CY = "#ff7f0e"
COLOR_CASSON = "#2ca02c"


@dataclass(frozen=True, slots=True)
class ModelState:
    name: str
    color: str
    model_obj: ConstitutiveModel
    model_factory: Callable[[float, float], ConstitutiveModel]
    nu_lbm: float
    n_vel: float
    nu_phys: float
    phys_dl: float
    phys_dt: float
    phys_dt_fn: Callable[[float], float]
    a_lbm: float
    a_phys: float
    w_fn: Callable[[float], float]
    w_newtonian_fn: Callable[[float], float]
    gamma_wall_phys: float
    eta_phys_fn: Callable[[float], float]
    is_powerlaw: bool


def compute_model_state(
    name: str,
    color: str,
    model_obj: ConstitutiveModel,
    model_factory: Callable[[float, float], ConstitutiveModel],
    nu_lbm: float,
    n_vel: float,
    nu_phys: float,
    phys_height: float,
    a_phys_val: float | None,
    u_max_phys_val: float | None,
    re_target: float | None,
    is_powerlaw: bool,
    phys_dt_fn: Callable[[float], float],
    re_ref_target: float | None = None,
) -> ModelState:
    res = 1
    z = BLOCK_SIZE * res
    n_fluid = z - 4
    r_lbm = n_fluid / 2.0
    phys_dl = phys_height / n_fluid
    phys_dt = phys_dt_fn(phys_dl)

    flux_fn = model_obj.flux
    pot_fn = model_obj.pot

    if a_phys_val is not None:
        a_lbm = a_phys_val * phys_dt**2 / phys_dl
    elif is_powerlaw:
        if u_max_phys_val is not None:
            u_max_phys = u_max_phys_val
        elif re_ref_target is not None:
            r_phys = phys_height / 2.0
            nu_ref_phys = nu_lbm * phys_dl**2 / phys_dt
            u_max_phys = re_ref_target * nu_ref_phys / (2.0 * r_phys)
        else:
            assert re_target is not None
            r_phys = phys_height / 2.0
            u_max_phys = _u_max_from_re_wall(
                re_target,
                r_phys,
                r_lbm,
                nu_lbm,
                phys_dl,
                phys_dt,
                n_vel,
                model_obj.nu,
                flux_fn,
                pot_fn,
            )
        u_max_lbm = u_max_phys * phys_dt / phys_dl
        exponent = (n_vel + 1.0) / n_vel
        K_lbm_model = flux_fn(1.0)
        a_lbm = (
            K_lbm_model
            * (u_max_lbm * (n_vel + 1.0) / n_vel * r_lbm ** (-exponent)) ** n_vel
        )
    else:
        if u_max_phys_val is not None:
            u_max_phys = u_max_phys_val
        elif re_ref_target is not None:
            r_phys = phys_height / 2.0
            nu_ref_phys = nu_lbm * phys_dl**2 / phys_dt
            u_max_phys = re_ref_target * nu_ref_phys / (2.0 * r_phys)
        else:
            assert re_target is not None
            r_phys = phys_height / 2.0
            u_max_phys = _u_max_from_re_wall(
                re_target,
                r_phys,
                r_lbm,
                nu_lbm,
                phys_dl,
                phys_dt,
                n_vel,
                model_obj.nu,
                flux_fn,
                pot_fn,
            )
        u_max_lbm = u_max_phys * phys_dt / phys_dl
        a_lbm = invert_A_lbm_from_umax(u_max_lbm, r_lbm, flux_fn, pot_fn)

    a_phys = a_lbm * phys_dl / phys_dt**2

    if is_powerlaw:

        def w_fn_pl(
            s_val: float,
            _r=r_lbm,
            _a=a_lbm,
            _K=flux_fn(1.0),
            _n=n_vel,
            _dl=phys_dl,
            _dt=phys_dt,
        ) -> float:
            s_lbm = s_val / _dl
            w_lbm = powerlaw_velocity(s_lbm, _r, _a, _K, _n)
            return w_lbm * _dl / _dt

        w_fn = w_fn_pl
    else:

        def w_fn_general(
            s_val: float,
            _r=r_lbm,
            _a=a_lbm,
            _f=flux_fn,
            _p=pot_fn,
            _dl=phys_dl,
            _dt=phys_dt,
        ) -> float:
            s_lbm = s_val / _dl
            w_lbm = poiseuille_W(s_lbm, _r, _a, _f, _p)
            return w_lbm * _dl / _dt

        w_fn = w_fn_general

    u_max_lbm_val = poiseuille_W(0.0, r_lbm, a_lbm, flux_fn, pot_fn)

    def w_newtonian_fn(
        s_val: float, _u=u_max_lbm_val, _r=r_lbm, _dl=phys_dl, _dt=phys_dt
    ) -> float:
        s_lbm = s_val / _dl
        w_lbm = _u * (_r**2 - s_lbm**2) / _r**2
        return w_lbm * _dl / _dt

    gamma_wall_lbm = solve_monotone(flux_fn, a_lbm * r_lbm)
    if gamma_wall_lbm <= 0:
        gamma_wall_lbm = 1.0
    gamma_wall_phys = gamma_wall_lbm / phys_dt

    visc_scale = phys_dl**2 / phys_dt
    nu_phys_fn = lambda g, _m=model_obj, _dt=phys_dt, _vs=visc_scale: (
        _m.nu(g * _dt) * _vs
    )

    return ModelState(
        name=name,
        color=color,
        model_obj=model_obj,
        model_factory=model_factory,
        nu_lbm=nu_lbm,
        n_vel=n_vel,
        nu_phys=nu_phys,
        phys_dl=phys_dl,
        phys_dt=phys_dt,
        phys_dt_fn=phys_dt_fn,
        a_lbm=a_lbm,
        a_phys=a_phys,
        w_fn=w_fn,
        w_newtonian_fn=w_newtonian_fn,
        gamma_wall_phys=gamma_wall_phys,
        eta_phys_fn=nu_phys_fn,
        is_powerlaw=is_powerlaw,
    )


# --------------------------------------------------------------------------- #
# Streamlit UI
# --------------------------------------------------------------------------- #


def main() -> None:
    if st.session_state.get("_pending_import"):
        data = st.session_state.pop("_pending_import")
        for k, v in data.items():
            st.session_state[k] = v
        st.rerun()

    st.set_page_config(
        page_title="Non-Newtonian Poiseuille Explorer",
        page_icon="🧪",
        layout="wide",
    )
    st.title("Non-Newtonian Poiseuille flow explorer")
    st.caption(
        "Power-law, Carreau-Yasuda (CY), and Casson constitutive models "
        "with analytical velocity profiles and dimensionless quantity verification."
    )

    st.sidebar.header("Flow Parameters")

    st.sidebar.number_input(
        "LBM viscosity ν_lbm",
        min_value=1e-6,
        max_value=0.166,
        format="%.6f",
        step=0.001,
        key="lbm_viscosity",
        help=(
            "Reference lattice viscosity for the collision (must be ≤ 1/6). "
            "Same for all models."
        ),
    )
    phys_height = st.sidebar.number_input(
        "Channel height H [m]",
        min_value=0.01,
        max_value=10.0,
        value=1.0,
        format="%.3f",
        step=0.1,
    )
    R_phys = phys_height / 2.0

    drive_mode = st.sidebar.segmented_control(
        "Driving parameter",
        ["Body force A [m/s²]", "Max velocity u_max [m/s]", "Re_wall", "Re_ref"],
        default="Body force A [m/s²]",
        required=True,
        key="drive_mode",
    )

    a_phys_val: float | None = None
    u_max_phys_val: float | None = None
    re_target: float | None = None
    re_ref_target: float | None = None

    if drive_mode.startswith("Body"):
        a_phys_val = st.sidebar.number_input(
            "A_phys [m/s²]",
            min_value=1e-12,
            max_value=1e4,
            value=1.0,
            format="%.3e",
            key="a_phys_input",
        )
    elif drive_mode.startswith("Max"):
        u_max_phys_val = st.sidebar.number_input(
            "u_max [m/s]",
            min_value=1e-12,
            max_value=100.0,
            value=0.6,
            format="%.3e",
            key="u_max_input",
        )
    elif drive_mode == "Re_wall":
        re_target = st.sidebar.number_input(
            "Re_wall",
            min_value=0.1,
            max_value=1e4,
            value=60.0,
            format="%.1f",
            key="re_input",
        )
    else:
        re_ref_target = st.sidebar.number_input(
            "Re_ref",
            min_value=0.1,
            max_value=1e4,
            value=60.0,
            format="%.1f",
            key="re_ref_input",
        )

    st.sidebar.divider()
    st.sidebar.header("Model Parameters")

    defaults = DEFAULTS
    for key, val in defaults.items():
        st.session_state.setdefault(key, val)

    def _make_models(params: dict | None = None):
        p = params if params is not None else st.session_state
        dl = phys_height / (BLOCK_SIZE - 4)
        nu_lbm = p["lbm_viscosity"]

        # Power-law: diffusive scaling with γ_ref = u_max/R
        # ν_ref = (K/ρ)·γ_ref^(n-1), dt = ν_lbm·dl²/ν_ref, K_lbm = (K/ρ)·dt^(2-n)/dl²
        n_pl_val = p["n_pl"]
        K_phys_val = p["K_phys"]
        R_phys_val = phys_height / 2.0
        u_max_pl = _pl_u_max_phys(
            p.get("drive_mode", "Body force A [m/s²]"),
            p.get("a_phys_input", 1.0),
            p.get("u_max_input", 0.6),
            p.get("re_input", 60.0),
            p.get("re_ref_input", 60.0),
            K_phys_val,
            n_pl_val,
            RHO,
            R_phys_val,
        )
        gamma_ref = u_max_pl / R_phys_val
        nu_ref_pl = (K_phys_val / RHO) * gamma_ref ** (n_pl_val - 1.0)
        phys_dt_fn_pl = lambda dl_, _nu=nu_ref_pl, _K=nu_lbm: _K * dl_**2 / _nu
        dt_pl = phys_dt_fn_pl(dl)
        K_lbm_pl = (K_phys_val / RHO) * dt_pl ** (2.0 - n_pl_val) / dl**2
        pl = PowerLaw(K=K_lbm_pl, n=n_pl_val)

        # CY
        eta_0_phys = p["eta_0"]
        eta_inf_phys = p["eta_inf"]
        nu_0_phys = eta_0_phys / RHO
        nu_inf_phys = eta_inf_phys / RHO
        phys_dt_fn_cy = lambda dl_, _K=nu_lbm, _n0=nu_0_phys: _K * dl_**2 / _n0
        nu_inf_lbm = nu_inf_phys * phys_dt_fn_cy(dl) / dl**2
        lambda_lbm = p["lambda_cy"] / phys_dt_fn_cy(dl)
        cy = CarreauYasuda(
            nu_inf=nu_inf_lbm,
            nu_0=nu_lbm,
            lambda_=lambda_lbm,
            a=p["a_cy"],
            n=p["n_cy"],
        )

        # Casson
        tau_y_phys = p["tau_y"]
        eta_C_phys = p["eta_C"]
        nu_C_phys = eta_C_phys / RHO
        phys_dt_fn_cas = lambda dl_, _K=nu_lbm, _nC=nu_C_phys: _K * dl_**2 / _nC
        k0_lbm = math.sqrt(tau_y_phys / RHO) * phys_dt_fn_cas(dl) / dl
        k1_lbm = math.sqrt(nu_lbm)
        cas = Casson(k0=k0_lbm, k1=k1_lbm)

        return (
            pl,
            cy,
            cas,
            phys_dt_fn_pl,
            phys_dt_fn_cy,
            phys_dt_fn_cas,
        )

    def _compute_source_u_max(model_idx: int, params: dict | None = None) -> float:
        """Compute u_max_phys for model at res=1 from current driving params.

        model_idx: 0=PL, 1=CY, 2=Casson.
        """
        p = params if params is not None else st.session_state
        nu_lbm = p["lbm_viscosity"]
        dl = phys_height / (BLOCK_SIZE - 4)
        n_fluid = BLOCK_SIZE - 4
        r_lbm = n_fluid / 2.0
        r_phys = phys_height / 2.0

        pl_obj, cy_obj, cas_obj, dt_fn_pl, dt_fn_cy, dt_fn_cas = _make_models(params)
        models = [
            (pl_obj, dt_fn_pl, p["n_pl"], True),
            (cy_obj, dt_fn_cy, p["n_cy"], False),
            (cas_obj, dt_fn_cas, 1.0, False),
        ]
        m_obj, dt_fn, n_vel, _is_pl = models[model_idx]
        dt = dt_fn(dl)

        drive = p.get("drive_mode", "Body force A [m/s²]")
        if drive.startswith("Body"):
            a_phys = p.get("a_phys_input", 1.0)
            a_lbm = a_phys * dt**2 / dl
        elif drive.startswith("Max"):
            return p.get("u_max_input", 0.6)
        elif drive == "Re_ref":
            nu_ref_phys = nu_lbm * dl**2 / dt
            re_ref_target = p.get("re_ref_input", 60.0)
            return re_ref_target * nu_ref_phys / (2.0 * r_phys)
        else:
            re_target = p.get("re_input", 60.0)
            return _u_max_from_re_wall(
                re_target,
                r_phys,
                r_lbm,
                nu_lbm,
                dl,
                dt,
                n_vel,
                m_obj.nu,
                m_obj.flux,
                m_obj.pot,
            )

        u_max_lbm = poiseuille_W(0.0, r_lbm, a_lbm, m_obj.flux, m_obj.pot)
        return u_max_lbm * dl / dt

    def _switch_to_u_max(source_u_max: float) -> None:
        """Switch driving mode to u_max and set value."""
        st.session_state["drive_mode"] = "Max velocity u_max [m/s]"
        st.session_state["u_max_input"] = source_u_max

    def _compute_source_a_phys(model_idx: int, params: dict | None = None) -> float:
        """Compute A_phys from current driving using the source model at res=1."""
        p = params if params is not None else st.session_state
        nu_lbm = p["lbm_viscosity"]
        dl = phys_height / (BLOCK_SIZE - 4)
        r_lbm = (BLOCK_SIZE - 4) / 2.0
        r_phys = phys_height / 2.0

        pl_obj, cy_obj, cas_obj, dt_fn_pl, dt_fn_cy, dt_fn_cas = _make_models(params)
        models = [
            (pl_obj, dt_fn_pl, p["n_pl"], True),
            (cy_obj, dt_fn_cy, p["n_cy"], False),
            (cas_obj, dt_fn_cas, 1.0, False),
        ]
        m_obj, dt_fn, n_vel, _ = models[model_idx]
        dt = dt_fn(dl)

        drive = p.get("drive_mode", "Body force A [m/s²]")
        if drive.startswith("Body"):
            return p.get("a_phys_input", 1.0)
        elif drive.startswith("Max"):
            u_max_phys = p.get("u_max_input", 0.6)
            u_max_lbm = u_max_phys * dt / dl
            a_lbm = invert_A_lbm_from_umax(u_max_lbm, r_lbm, m_obj.flux, m_obj.pot)
            return a_lbm * dl / dt**2
        elif drive == "Re_ref":
            nu_ref_phys = nu_lbm * dl**2 / dt
            re_ref_target = p.get("re_ref_input", 60.0)
            u_max_phys = re_ref_target * nu_ref_phys / (2.0 * r_phys)
            u_max_lbm = u_max_phys * dt / dl
            a_lbm = invert_A_lbm_from_umax(u_max_lbm, r_lbm, m_obj.flux, m_obj.pot)
            return a_lbm * dl / dt**2
        else:
            re_target = p.get("re_input", 60.0)
            u_max_phys = _u_max_from_re_wall(
                re_target,
                r_phys,
                r_lbm,
                nu_lbm,
                dl,
                dt,
                n_vel,
                m_obj.nu,
                m_obj.flux,
                m_obj.pot,
            )
            u_max_lbm = u_max_phys * dt / dl
            a_lbm = invert_A_lbm_from_umax(u_max_lbm, r_lbm, m_obj.flux, m_obj.pot)
            return a_lbm * dl / dt**2

    def _fit_pl_from_cy():
        p = st.session_state
        for k in MODEL_KEYS["pl"]:
            p[k] = DEFAULTS[k]
        dl = phys_height / (BLOCK_SIZE - 4)
        _, cy, _, _, dt_fn_cy, _ = _make_models()
        dt_src = dt_fn_cy(dl)
        a_phys = _compute_source_a_phys(1)
        nu_lbm = p["lbm_viscosity"]
        K_phys = DEFAULTS["K_phys"]
        R_phys_fit = phys_height / 2.0

        def pl_builder(params):
            n = float(params[0])
            u_max = (
                (a_phys * RHO / K_phys) ** (1.0 / n)
                * n
                / (n + 1.0)
                * R_phys_fit ** ((n + 1.0) / n)
            )
            nu_ref = (K_phys / RHO) * (u_max / R_phys_fit) ** (n - 1.0)
            dt = nu_lbm * dl**2 / nu_ref
            K_lbm_model = (K_phys / RHO) * dt ** (2.0 - n) / dl**2
            return PowerLaw(K=K_lbm_model, n=n), dt

        result = fit_by_velocity_profile(
            cy, dt_src, pl_builder, [(0.1, 3.0)], a_phys, phys_height, dl
        )
        if result is not None:
            p["n_pl"] = float(result[0])
        _switch_to_u_max(_compute_source_u_max(1))

    def _fit_pl_from_cas():
        p = st.session_state
        for k in MODEL_KEYS["pl"]:
            p[k] = DEFAULTS[k]
        dl = phys_height / (BLOCK_SIZE - 4)
        _, _, cas, _, _, dt_fn_cas = _make_models()
        dt_src = dt_fn_cas(dl)
        a_phys = _compute_source_a_phys(2)
        nu_lbm = p["lbm_viscosity"]
        K_phys = DEFAULTS["K_phys"]
        R_phys_fit = phys_height / 2.0

        def pl_builder(params):
            n = float(params[0])
            u_max = (
                (a_phys * RHO / K_phys) ** (1.0 / n)
                * n
                / (n + 1.0)
                * R_phys_fit ** ((n + 1.0) / n)
            )
            nu_ref = (K_phys / RHO) * (u_max / R_phys_fit) ** (n - 1.0)
            dt = nu_lbm * dl**2 / nu_ref
            K_lbm_model = (K_phys / RHO) * dt ** (2.0 - n) / dl**2
            return PowerLaw(K=K_lbm_model, n=n), dt

        result = fit_by_velocity_profile(
            cas, dt_src, pl_builder, [(0.1, 3.0)], a_phys, phys_height, dl
        )
        if result is not None:
            p["n_pl"] = float(result[0])
        _switch_to_u_max(_compute_source_u_max(2))

    def _fit_cy_from_pl():
        p = st.session_state
        for k in MODEL_KEYS["cy"]:
            p[k] = DEFAULTS[k]
        dl = phys_height / (BLOCK_SIZE - 4)
        pl, _, _, dt_fn_pl, _, _ = _make_models()
        dt_src = dt_fn_pl(dl)
        a_phys = _compute_source_a_phys(0)
        nu_lbm = p["lbm_viscosity"]
        eta_0 = DEFAULTS["eta_0"]
        nu_0_phys = eta_0 / RHO
        dt_cy = nu_lbm * dl**2 / nu_0_phys

        def cy_builder(params):
            eta_inf, lambda_cy, a_cy, n_cy = (
                float(params[0]),
                float(params[1]),
                float(params[2]),
                float(params[3]),
            )
            nu_inf_lbm = (eta_inf / RHO) * dt_cy / dl**2
            m = CarreauYasuda(
                nu_inf=nu_inf_lbm,
                nu_0=nu_lbm,
                lambda_=lambda_cy / dt_cy,
                a=a_cy,
                n=n_cy,
            )
            return m, dt_cy

        bounds = [(1e-12, eta_0), (1e-12, 1e6), (0.1, 5.0), (0.1, 3.0)]
        result = fit_by_velocity_profile(
            pl, dt_src, cy_builder, bounds, a_phys, phys_height, dl
        )
        if result is not None:
            p["eta_inf"] = float(result[0])
            p["lambda_cy"] = float(result[1])
            p["a_cy"] = float(result[2])
            p["n_cy"] = float(result[3])
        _switch_to_u_max(_compute_source_u_max(0))

    def _fit_cy_from_cas():
        p = st.session_state
        for k in MODEL_KEYS["cy"]:
            p[k] = DEFAULTS[k]
        dl = phys_height / (BLOCK_SIZE - 4)
        _, _, cas, _, _, dt_fn_cas = _make_models()
        dt_src = dt_fn_cas(dl)
        a_phys = _compute_source_a_phys(2)
        nu_lbm = p["lbm_viscosity"]
        eta_0 = DEFAULTS["eta_0"]
        nu_0_phys = eta_0 / RHO
        dt_cy = nu_lbm * dl**2 / nu_0_phys

        def cy_builder(params):
            eta_inf, lambda_cy, a_cy, n_cy = (
                float(params[0]),
                float(params[1]),
                float(params[2]),
                float(params[3]),
            )
            nu_inf_lbm = (eta_inf / RHO) * dt_cy / dl**2
            m = CarreauYasuda(
                nu_inf=nu_inf_lbm,
                nu_0=nu_lbm,
                lambda_=lambda_cy / dt_cy,
                a=a_cy,
                n=n_cy,
            )
            return m, dt_cy

        bounds = [(1e-12, eta_0), (1e-12, 1e6), (0.1, 5.0), (0.1, 3.0)]
        result = fit_by_velocity_profile(
            cas, dt_src, cy_builder, bounds, a_phys, phys_height, dl
        )
        if result is not None:
            p["eta_inf"] = float(result[0])
            p["lambda_cy"] = float(result[1])
            p["a_cy"] = float(result[2])
            p["n_cy"] = float(result[3])
        _switch_to_u_max(_compute_source_u_max(2))

    def _fit_cas_from_pl():
        p = st.session_state
        for k in MODEL_KEYS["cas"]:
            p[k] = DEFAULTS[k]
        dl = phys_height / (BLOCK_SIZE - 4)
        r_phys = phys_height / 2.0
        r_lbm = (BLOCK_SIZE - 4) / 2.0
        pl, _, _, dt_fn_pl, _, _ = _make_models()
        dt_src = dt_fn_pl(dl)
        a_phys = _compute_source_a_phys(0)
        nu_lbm = p["lbm_viscosity"]

        a_lbm_src = a_phys * dt_src**2 / dl
        gamma_wall_src = solve_monotone(pl.flux, a_lbm_src * r_lbm)
        nu_wall_src = pl.nu(gamma_wall_src) if gamma_wall_src > 0 else nu_lbm
        eta_wall_src = nu_wall_src * dl**2 / dt_src * RHO
        tau_y_max = a_phys * r_phys * RHO * 0.99

        def cas_builder(params):
            tau_y = float(params[0])
            eta_C_val = float(params[1])
            nu_C_phys = eta_C_val / RHO
            dt_cas = nu_lbm * dl**2 / nu_C_phys
            m = Casson(k0=math.sqrt(tau_y / RHO) * dt_cas / dl, k1=math.sqrt(nu_lbm))
            return m, dt_cas

        result = fit_by_velocity_profile(
            pl,
            dt_src,
            cas_builder,
            [(0.0, tau_y_max), (1e-12, eta_wall_src)],
            a_phys,
            phys_height,
            dl,
        )
        if result is not None:
            p["tau_y"] = float(result[0])
            p["eta_C"] = float(result[1])
        _switch_to_u_max(_compute_source_u_max(0))

    def _fit_cas_from_cy():
        p = st.session_state
        for k in MODEL_KEYS["cas"]:
            p[k] = DEFAULTS[k]
        dl = phys_height / (BLOCK_SIZE - 4)
        r_phys = phys_height / 2.0
        r_lbm = (BLOCK_SIZE - 4) / 2.0
        _, cy, _, _, dt_fn_cy, _ = _make_models()
        dt_src = dt_fn_cy(dl)
        a_phys = _compute_source_a_phys(1)
        nu_lbm = p["lbm_viscosity"]

        a_lbm_src = a_phys * dt_src**2 / dl
        gamma_wall_src = solve_monotone(cy.flux, a_lbm_src * r_lbm)
        nu_wall_src = cy.nu(gamma_wall_src) if gamma_wall_src > 0 else nu_lbm
        eta_wall_src = nu_wall_src * dl**2 / dt_src * RHO
        tau_y_max = a_phys * r_phys * RHO * 0.99

        def cas_builder(params):
            tau_y = float(params[0])
            eta_C_val = float(params[1])
            nu_C_phys = eta_C_val / RHO
            dt_cas = nu_lbm * dl**2 / nu_C_phys
            m = Casson(k0=math.sqrt(tau_y / RHO) * dt_cas / dl, k1=math.sqrt(nu_lbm))
            return m, dt_cas

        result = fit_by_velocity_profile(
            cy,
            dt_src,
            cas_builder,
            [(0.0, tau_y_max), (1e-12, eta_wall_src)],
            a_phys,
            phys_height,
            dl,
        )
        if result is not None:
            p["tau_y"] = float(result[0])
            p["eta_C"] = float(result[1])
        _switch_to_u_max(_compute_source_u_max(1))

    with st.sidebar.expander("Power-law", expanded=True):
        st.number_input(
            "K_phys [Pa·s^n]",
            min_value=1e-12,
            max_value=100.0,
            format="%.3e",
            step=0.01,
            key="K_phys",
        )
        st.number_input(
            "Power-law index n",
            min_value=0.1,
            max_value=3.0,
            format="%.2f",
            step=0.1,
            key="n_pl",
            help="n<1 shear-thinning, n=1 Newtonian, n>1 shear-thickening.",
        )
        with st.container(horizontal=True):
            st.button("Fit from CY", key="fit_pl_cy", on_click=_fit_pl_from_cy)
            st.button("Fit from Casson", key="fit_pl_cas", on_click=_fit_pl_from_cas)

    with st.sidebar.expander("Carreau-Yasuda", expanded=True):
        st.number_input(
            "η_inf [Pa·s]",
            min_value=1e-12,
            max_value=100.0,
            format="%.3e",
            step=0.01,
            key="eta_inf",
        )
        st.number_input(
            "η₀ [Pa·s]",
            min_value=1e-12,
            max_value=100.0,
            format="%.3e",
            step=0.01,
            key="eta_0",
        )
        st.number_input(
            "λ [s]",
            min_value=1e-12,
            max_value=1e6,
            format="%.3e",
            key="lambda_cy",
        )
        st.number_input(
            "a (Yasuda exponent)",
            min_value=0.1,
            max_value=5.0,
            format="%.2f",
            key="a_cy",
        )
        st.number_input(
            "Power-law index n",
            min_value=0.1,
            max_value=3.0,
            format="%.2f",
            key="n_cy",
        )
        with st.container(horizontal=True):
            st.button("Fit from Power-law", key="fit_cy_pl", on_click=_fit_cy_from_pl)
            st.button("Fit from Casson", key="fit_cy_cas", on_click=_fit_cy_from_cas)

    with st.sidebar.expander("Casson", expanded=True):
        st.number_input(
            "τ_y [Pa]",
            min_value=0.0,
            max_value=100.0,
            format="%.3e",
            key="tau_y",
        )
        st.number_input(
            "η_C [Pa·s]",
            min_value=1e-12,
            max_value=100.0,
            format="%.3e",
            key="eta_C",
        )
        with st.container(horizontal=True):
            st.button("Fit from Power-law", key="fit_cas_pl", on_click=_fit_cas_from_pl)
            st.button("Fit from CY", key="fit_cas_cy", on_click=_fit_cas_from_cy)

    with st.sidebar.expander("Session Parameters", expanded=False):
        export_keys = [
            "lbm_viscosity",
            "n_pl",
            "K_phys",
            "eta_inf",
            "eta_0",
            "lambda_cy",
            "a_cy",
            "n_cy",
            "tau_y",
            "eta_C",
            "drive_mode",
            "a_phys_input",
            "u_max_input",
            "re_input",
            "re_ref_input",
        ]
        col_exp, col_imp = st.columns(2)
        with col_exp:
            if st.button("Export"):
                export_data = {
                    k: st.session_state.get(k, DEFAULTS.get(k)) for k in export_keys
                }
                st.session_state["_export_text"] = json.dumps(export_data, indent=2)
        with col_imp:
            import_clicked = st.button("Import")

        st.text_area(
            "Parameters (JSON)",
            value=st.session_state.get("_export_text", ""),
            height=150,
            key="_import_text",
        )

        if import_clicked:
            try:
                data = json.loads(st.session_state.get("_import_text", "") or "")
                st.session_state["_pending_import"] = data
                st.rerun()
            except json.JSONDecodeError as e:
                st.error(f"Invalid JSON: {e}")

        if st.session_state.get("_export_text"):
            st.code(st.session_state["_export_text"], language="json")

    lbm_viscosity = st.session_state["lbm_viscosity"]
    n_pl = st.session_state["n_pl"]
    K_phys = st.session_state["K_phys"]
    eta_inf = st.session_state["eta_inf"]
    eta_0 = st.session_state["eta_0"]
    lambda_cy_phys = st.session_state["lambda_cy"]
    a_cy = st.session_state["a_cy"]
    n_cy = st.session_state["n_cy"]
    tau_y = st.session_state["tau_y"]
    eta_C = st.session_state["eta_C"]

    res_demo = 1
    z_demo = BLOCK_SIZE * res_demo
    n_fluid_demo = z_demo - 4
    phys_dl_demo = phys_height / n_fluid_demo

    pl_obj, cy_obj, cas_obj, phys_dt_fn_pl, phys_dt_fn_cy, phys_dt_fn_cas = (
        _make_models()
    )

    nu_phys_pl = lbm_viscosity * phys_dl_demo**2 / phys_dt_fn_pl(phys_dl_demo)
    nu_phys_cy = eta_0 / RHO
    nu_phys_cas = eta_C / RHO

    # Per-resolution model factories: reconstruct model with correct LBM params
    # at each (dl, dt). Power-law K_lbm varies with resolution; CY lambda_lbm and
    # Casson k0_lbm must be recomputed (they scale as 1/dt and dt/dl respectively).
    def _pl_factory(dl, dt):
        K_lbm_res = (K_phys / RHO) * dt ** (2.0 - n_pl) / dl**2
        return PowerLaw(K=K_lbm_res, n=n_pl)

    pl_factory = _pl_factory

    def _cy_factory(dl, dt):
        nu_inf_lbm = (eta_inf / RHO) * dt / dl**2
        return CarreauYasuda(
            nu_inf=nu_inf_lbm,
            nu_0=lbm_viscosity,
            lambda_=lambda_cy_phys / dt,
            a=a_cy,
            n=n_cy,
        )

    def _cas_factory(dl, dt):
        return Casson(
            k0=math.sqrt(tau_y / RHO) * dt / dl,
            k1=math.sqrt(lbm_viscosity),
        )

    cy_factory = _cy_factory
    cas_factory = _cas_factory

    R_phys_val = phys_height / 2.0
    u_max_pl = _pl_u_max_phys(
        st.session_state.get("drive_mode", "Body force A [m/s²]"),
        a_phys_val,
        u_max_phys_val,
        re_target,
        re_ref_target,
        K_phys,
        n_pl,
        RHO,
        R_phys_val,
    )

    st_pl = compute_model_state(
        "Power-law",
        COLOR_POWERLAW,
        pl_obj,
        pl_factory,
        lbm_viscosity,
        n_pl,
        nu_phys_pl,
        phys_height,
        None,
        u_max_pl,
        None,
        is_powerlaw=True,
        phys_dt_fn=phys_dt_fn_pl,
        re_ref_target=None,
    )
    st_cy = compute_model_state(
        "Carreau-Yasuda",
        COLOR_CY,
        cy_obj,
        cy_factory,
        lbm_viscosity,
        n_cy,
        nu_phys_cy,
        phys_height,
        a_phys_val,
        u_max_phys_val,
        re_target,
        is_powerlaw=False,
        phys_dt_fn=phys_dt_fn_cy,
        re_ref_target=re_ref_target,
    )
    st_cas = compute_model_state(
        "Casson",
        COLOR_CASSON,
        cas_obj,
        cas_factory,
        lbm_viscosity,
        1.0,
        nu_phys_cas,
        phys_height,
        a_phys_val,
        u_max_phys_val,
        re_target,
        is_powerlaw=False,
        phys_dt_fn=phys_dt_fn_cas,
        re_ref_target=re_ref_target,
    )

    all_states = [st_pl, st_cy, st_cas]

    # --- Main content: two plots side by side ---
    col_visc, col_vel = st.columns(2)

    with col_visc:
        st.subheader("Viscosity vs Shear rate")

        visc_models_dynamic = []
        for s_obj in all_states:

            def make_dyn_fn(state_obj):
                return lambda g: state_obj.eta_phys_fn(g) * RHO

            visc_models_dynamic.append((s_obj.name, make_dyn_fn(s_obj), s_obj.color))

        fig_visc = plot_viscosity_multi(
            visc_models_dynamic, gamma_min=1e-2, gamma_max=1e2
        )
        st.plotly_chart(fig_visc)
        wall_rates = ", ".join(f"{s.name}: {s.gamma_wall_phys:.4g}" for s in all_states)
        st.caption(f"Wall shear rate γ̇_w at res=1 — {wall_rates} [1/s]")

    with col_vel:
        st.subheader("Velocity profile")
        show_newtonian = st.session_state.get("newtonian_overlay", True)
        vel_models = [(s.name, s.w_fn, s.color) for s in all_states]
        newt_models = (
            [("", all_states[0].w_newtonian_fn, "#d62728")] if show_newtonian else None
        )
        fig_vel = plot_velocity_multi(vel_models, newt_models, R_phys)
        st.plotly_chart(fig_vel)
        st.checkbox("Overlay Newtonian parabola", value=True, key="newtonian_overlay")

    # --- Dimensionless quantities ---
    st.divider()
    st.subheader("Dimensionless quantities")
    st.caption(
        "Verification across resolutions. "
        "Constraints: Ma < 0.1 (incompressible), ω ∈ [0.5, 1.9] (stable), "
        "Re_wall constant across resolutions."
    )

    resolutions = st.multiselect(
        "Resolutions to display",
        options=[1, 2, 3, 4, 8],
        default=[1, 2, 4, 8],
    )

    all_rows: list[dict] = []
    grid_rows: list[dict] = []
    for state in all_states:
        for res in sorted(resolutions):
            n_fluid_res = BLOCK_SIZE * res - 4
            phys_dl_res = phys_height / n_fluid_res
            dt_res = state.phys_dt_fn(phys_dl_res)
            model_res = state.model_factory(phys_dl_res, dt_res)
            if state.is_powerlaw:
                row_a, row_u, row_re, row_ref = None, u_max_pl, None, None
            else:
                row_a, row_u, row_re, row_ref = (
                    a_phys_val,
                    u_max_phys_val,
                    re_target,
                    re_ref_target,
                )
            row = compute_dimensionless(
                nu_lbm=state.nu_lbm,
                nu_phys=state.nu_phys,
                n=state.n_vel,
                phys_height=phys_height,
                res=res,
                a_phys_val=row_a,
                u_max_phys_val=row_u,
                re_target=row_re,
                re_ref_target=row_ref,
                nu_fn=model_res.nu,
                flux_fn=model_res.flux,
                pot_fn=model_res.pot,
                is_powerlaw=state.is_powerlaw,
                phys_dt_fn=state.phys_dt_fn,
            )

            all_rows.append(
                {
                    "Model": state.name,
                    "Res": row.res,
                    "dt": row.phys_dt,
                    "u_max_lbm": row.u_max_lbm,
                    "Ma": row.ma,
                    "u_max_phys": row.u_max_phys,
                    "Re_wall": row.re,
                    "Re_ref": row.re_ref,
                    "ν_wall": row.nu_wall,
                    "ν_ratio": row.nu_ratio,
                    "A_lbm": row.a_lbm,
                    "A_phys": row.a_phys,
                    "t_steady_phys": row.t_steady_phys,
                }
            )
            if state is all_states[0]:
                grid_rows.append(
                    {
                        "Res": row.res,
                        "N_fluid": row.n_fluid,
                        "R_lbm": row.r_lbm,
                        "dx": row.phys_dl,
                        "ω": row.omega,
                        "t_steady_lbm": row.t_steady_lbm,
                    }
                )

    st.markdown("**Grid parameters** (model-independent)")
    st.dataframe(pd.DataFrame(grid_rows).style.format(precision=4))
    st.caption(
        "dt is model-dependent — each model defines its own dt so that "
        "ν_lbm is the collision reference viscosity. See model-specific table below."
    )

    st.markdown("**Model-specific quantities**")
    st.dataframe(
        pd.DataFrame(all_rows).style.format(
            {"A_lbm": "{:g}", "A_phys": "{:g}"}, precision=4
        )
    )

    st.markdown("**Consistency checks:**")
    ma_vals = [r["Ma"] for r in all_rows]
    omega_val = grid_rows[0]["ω"] if grid_rows else 0.0
    re_vals = [r["Re_wall"] for r in all_rows]

    ma_ok = "✅" if max(ma_vals) < MA_MAX else "❌"
    omega_ok = "✅" if OMEGA_MIN < omega_val < OMEGA_MAX else "❌"
    re_ok = "✅" if max(re_vals) / max(min(re_vals), 1e-12) < 1.5 else "❌"

    c1, c2, c3 = st.columns(3)
    c1.metric(f"{ma_ok} Ma < 0.1", value=f"{max(ma_vals):.4f}")
    c2.metric(f"{omega_ok} ω ∈ [0.5, 1.9]", value=f"{omega_val:.3f}")
    c3.metric(
        f"{re_ok} Re_wall constant", value=f"{min(re_vals):.1f}–{max(re_vals):.1f}"
    )

    with st.expander("Reynolds number definitions"):
        st.markdown(
            """
**Re_wall** — based on the actual viscosity at the wall shear rate
(`ν_wall = ν(γ̇_wall)`). This is the physical Reynolds number for the flow
and is resolution-invariant for all models and driving modes. Use this for
cross-model comparison.

**Re_ref** — based on each model's reference (collision) viscosity
(`ν_ref = ν_lbm · dl²/dt`). Matches the C++ sim's `Re` definition
(`PHYS_VISCOSITY`).
- **CY**: `ν_ref = η_0/ρ` (zero-shear viscosity) — invariant
- **Casson**: `ν_ref = η_C/ρ` (plastic viscosity) — invariant
- **Power-law**: `ν_ref = (K/ρ)·(u_max/R)^(n-1)` (viscosity at characteristic
  shear rate `γ_ref = u_max/R`) — invariant, equals the generalized
  Reynolds number `Re_gen = 2R²·u_max^(2-n)·ρ/K`
            """
        )

    with st.expander("How fitting works"):
        st.markdown(
            """
**Velocity-profile fitting** matches the normalized shape `W/u_max` at
30 points across the channel using `differential_evolution` (seed=42).

After fitting, the driving mode switches to **u_max** with the source
model's `u_max`. This preserves the flow rate and profile shape, but
**not** `Re_wall` or `Re_ref` — those depend on model-specific viscosities
(`ν_wall` or `ν_ref`) that differ between source and target.

**Parameters fitted:**
- **Power-law** (1 param): `n` — `K_phys` fixed (shape depends only on `n`)
- **CY** (4 params): `η_inf, λ, a, n` — `η₀` fixed (sets collision reference)
- **Casson** (2 params): `τ_y, η_C` — both vary freely

Target model parameters are reset to `DEFAULTS` before fitting.
`ν_lbm` is shared across all models and never reset.
            """
        )

    with st.expander("dt definitions (model-specific)"):
        st.markdown(
            """
Each model defines its own `dt` so that `ν_lbm` (the collision reference
viscosity) corresponds to a model-specific physical quantity. This makes `dt`
— and therefore `u_max_lbm`, `Re_ref`, and `t_steady_phys` — model-dependent
even when `u_max_phys` and `ν_lbm` are fixed across models.

All three models use **diffusive scaling** (`dt ∝ dl²`), keeping `ν_lbm`
invariant across resolutions.

- **Power-law**: `ν_ref = (K/ρ)·(u_max/R)^(n-1)` (viscosity at characteristic
  shear rate `γ_ref = u_max/R`). Then `dt = ν_lbm·dl²/ν_ref` (diffusive).
  `K_lbm = (K/ρ)·dt^(2-n)/dl²` varies with resolution for `n ≠ 1`, but
  `ν_lbm = K_lbm·γ_ref_lbm^(n-1)` stays invariant. For `n = 1`: `ν_ref = K/ρ`,
  `K_lbm = ν_lbm` (identical to Newtonian).
- **CY**: `dt = ν_lbm · dl² / (η₀/ρ)` — chosen so that `ν_0_lbm = ν_lbm`
  (zero-shear viscosity). `ν_ref = η₀/ρ`, resolution-invariant.
- **Casson**: `dt = ν_lbm · dl² / (η_C/ρ)` — chosen so that
  `k₁² = ν_lbm` (plastic viscosity). `ν_ref = η_C/ρ`, resolution-invariant.
            """
        )


if __name__ == "__main__":
    main()
