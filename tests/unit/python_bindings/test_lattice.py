"""Unit tests for the Lattice bindings of pytnl_lbm.

The Lattice class wraps the unit-conversion helpers from ``include/lbm3d/lattice.h``
(viscosity, coordinates, velocity, force) exported for all four
(precision, index type) combinations. These tests exercise the conversions
against exact formulas from the header; no simulation is started.
"""

from __future__ import annotations

import pytest

import pytnl_lbm

type AnyLattice = (
    pytnl_lbm.Lattice_3_float_int
    | pytnl_lbm.Lattice_3_float_long
    | pytnl_lbm.Lattice_3_double_int
    | pytnl_lbm.Lattice_3_double_long
)

LATTICE_CLASSES = [
    "Lattice_3_float_int",
    "Lattice_3_float_long",
    "Lattice_3_double_int",
    "Lattice_3_double_long",
]

PHYS_DL = 0.1
PHYS_DT = 0.005
PHYS_NU = 1.5e-5


def make_lattice(class_name: str) -> AnyLattice:
    lat = getattr(pytnl_lbm, class_name)()
    lat.physDl = PHYS_DL
    lat.physDt = PHYS_DT
    lat.physViscosity = PHYS_NU
    return lat


def float_tolerance(lattice: AnyLattice) -> float:
    prefix = type(lattice).__name__.split("_")[2]
    return 1e-5 if prefix == "float" else 1e-14


@pytest.fixture(params=LATTICE_CLASSES)
def lattice(request: pytest.FixtureRequest) -> AnyLattice:
    return make_lattice(request.param)


class TestLatticeBasics:
    @pytest.mark.parametrize("class_name", LATTICE_CLASSES)
    def test_dimension(self, class_name: str) -> None:
        lattice_cls = getattr(pytnl_lbm, class_name)
        assert lattice_cls().D == 3
        # getMeshDimension is a static method in the binding (class-level call)
        assert lattice_cls.getMeshDimension() == 3

    @pytest.mark.parametrize("class_name", LATTICE_CLASSES)
    def test_attribute_roundtrip(self, class_name: str) -> None:
        lat = make_lattice(class_name)
        assert lat.physDl == pytest.approx(PHYS_DL)
        assert lat.physDt == pytest.approx(PHYS_DT)
        assert lat.physViscosity == pytest.approx(PHYS_NU)


class TestViscosityConversion:
    def test_phys2lbm(self, lattice: AnyLattice) -> None:
        # phys2lbmViscosity = phys * dt / dl^2  (lattice.h:51)
        expected = PHYS_NU * PHYS_DT / (PHYS_DL * PHYS_DL)
        result = lattice.phys2lbmViscosity(PHYS_NU)
        assert result == pytest.approx(expected, abs=float_tolerance(lattice))

    def test_lbmViscosity_uses_attribute(self, lattice: AnyLattice) -> None:
        assert lattice.lbmViscosity() == pytest.approx(
            lattice.phys2lbmViscosity(PHYS_NU), abs=float_tolerance(lattice)
        )

    def test_roundtrip(self, lattice: AnyLattice) -> None:
        lbm_nu = lattice.phys2lbmViscosity(PHYS_NU)
        assert lattice.lbm2physViscosity(lbm_nu) == pytest.approx(
            PHYS_NU, abs=float_tolerance(lattice)
        )


class TestCoordinateConversion:
    def test_lbm2phys_centered_coordinates(self, lattice: AnyLattice) -> None:
        # lbm2physX(x) = physOrigin.x + (x - 0.5) * dl  (lattice.h:71)
        assert lattice.lbm2physX(4) == pytest.approx(0.35, abs=float_tolerance(lattice))
        assert lattice.lbm2physY(3) == pytest.approx(0.25, abs=float_tolerance(lattice))
        assert lattice.lbm2physZ(1) == pytest.approx(0.05, abs=float_tolerance(lattice))

    def test_phys2lbm_interpolation(self, lattice: AnyLattice) -> None:
        # phys2lbmX(x) = (x - physOrigin.x) / dl + 0.5  (lattice.h:89)
        assert lattice.phys2lbmX(0.35) == pytest.approx(
            4.0, abs=float_tolerance(lattice)
        )
        assert lattice.phys2lbmY(0.25) == pytest.approx(
            3.0, abs=float_tolerance(lattice)
        )
        assert lattice.phys2lbmZ(0.05) == pytest.approx(
            1.0, abs=float_tolerance(lattice)
        )

    @pytest.mark.parametrize("index", [1, 7, 42])
    def test_roundtrip_at_lattice_points(self, lattice: AnyLattice, index: int) -> None:
        assert lattice.phys2lbmX(lattice.lbm2physX(index)) == pytest.approx(
            index, abs=float_tolerance(lattice)
        )
        assert lattice.phys2lbmY(lattice.lbm2physY(index)) == pytest.approx(
            index, abs=float_tolerance(lattice)
        )
        assert lattice.phys2lbmZ(lattice.lbm2physZ(index)) == pytest.approx(
            index, abs=float_tolerance(lattice)
        )


class TestVelocityConversion:
    def test_phys2lbm(self, lattice: AnyLattice) -> None:
        # phys2lbmVelocity = u * dt / dl  (lattice.h:106)
        expected = 0.02 * PHYS_DT / PHYS_DL
        result = lattice.phys2lbmVelocity(0.02)
        assert result == pytest.approx(expected, abs=float_tolerance(lattice))

    def test_lbm2phys(self, lattice: AnyLattice) -> None:
        # lbm2physVelocity = u * dl / dt  (lattice.h:102)
        expected = 0.001 * PHYS_DL / PHYS_DT
        result = lattice.lbm2physVelocity(0.001)
        assert result == pytest.approx(expected, abs=float_tolerance(lattice))

    def test_roundtrip(self, lattice: AnyLattice) -> None:
        u = 0.02
        assert lattice.phys2lbmVelocity(lattice.lbm2physVelocity(u)) == pytest.approx(
            u, abs=float_tolerance(lattice)
        )


class TestForceConversion:
    def test_phys2lbm(self, lattice: AnyLattice) -> None:
        # phys2lbmForce = f * dt^2 / dl  (lattice.h:115)
        expected = 0.001 * PHYS_DT * PHYS_DT / PHYS_DL
        result = lattice.phys2lbmForce(0.001)
        assert result == pytest.approx(expected, abs=float_tolerance(lattice))

    def test_lbm2phys(self, lattice: AnyLattice) -> None:
        # lbm2physForce = f * dl / dt^2  (lattice.h:111)
        expected = 2.5e-4 * PHYS_DL / (PHYS_DT * PHYS_DT)
        result = lattice.lbm2physForce(2.5e-4)
        assert result == pytest.approx(expected, abs=float_tolerance(lattice))

    def test_roundtrip(self, lattice: AnyLattice) -> None:
        force = 0.001
        assert lattice.phys2lbmForce(lattice.lbm2physForce(force)) == pytest.approx(
            force, abs=float_tolerance(lattice)
        )
