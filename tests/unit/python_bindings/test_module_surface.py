"""Smoke tests for the overall pytnl_lbm module surface.

Pins the exported names the bindings promise (the build covers exactly one
concrete SP_D3Q27_CUM_ConstInflow instantiation plus the generic helpers), so
renames or accidental export losses fail loudly.
"""

from __future__ import annotations

import pytest

import pytnl_lbm

EXPECTED_EXPORTS = [
    "execute",
    "getMacroView",
    "UniformDataWriter",
    "Actions",
    "counter_float",
    "counter_double",
    "Lattice_3_float_int",
    "Lattice_3_float_long",
    "Lattice_3_double_int",
    "Lattice_3_double_long",
    "LBM_SP_D3Q27_CUM_ConstInflow",
    "LBM_BLOCK_SP_D3Q27_CUM_ConstInflow",
    "LBM_Data_SP_D3Q27_CUM_ConstInflow",
    "State_SP_D3Q27_CUM_ConstInflow",
    "hmacro_array_SP_D3Q27_CUM_ConstInflow",
    "dmacro_array_SP_D3Q27_CUM_ConstInflow",
    "dist_hmacro_array_SP_D3Q27_CUM_ConstInflow",
    "dist_dmacro_array_SP_D3Q27_CUM_ConstInflow",
    "hmacro_view_SP_D3Q27_CUM_ConstInflow",
    "dmacro_view_SP_D3Q27_CUM_ConstInflow",
    "macro_indexer_SP_D3Q27_CUM_ConstInflow",
]

ACTION_VALUES = {
    "STAT_RESET": 0,
    "STAT2_RESET": 1,
    "PRINT": 2,
    "OUT2D": 3,
    "OUT3D": 4,
    "OUT3DCUT": 5,
    "PROBE1": 6,
    "PROBE2": 7,
    "PROBE3": 8,
    "SAVESTATE": 9,
}


@pytest.mark.parametrize("name", EXPECTED_EXPORTS)
def test_export_present(name: str) -> None:
    assert hasattr(pytnl_lbm, name), f"pytnl_lbm no longer exports {name}"


def test_execute_and_macro_view_callable() -> None:
    assert callable(pytnl_lbm.execute)
    assert callable(pytnl_lbm.getMacroView)


@pytest.mark.parametrize(("name", "expected_value"), ACTION_VALUES.items())
def test_action_enum_values(name: str, expected_value: int) -> None:
    assert hasattr(pytnl_lbm.Actions, name)
    assert int(getattr(pytnl_lbm.Actions, name)) == expected_value


def test_action_enum_values_distinct() -> None:
    values = {int(getattr(pytnl_lbm.Actions, name)) for name in ACTION_VALUES}
    assert len(values) == len(ACTION_VALUES)
