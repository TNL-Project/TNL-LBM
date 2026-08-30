add_subdirectory(integration)
add_subdirectory(unit)
add_subdirectory(regression)

# The AMR doctest suites (coupling/subcycling/vtkhdf_writer/nesting, the C2F
# debug-define smokes, and the F2C Schönherr exactness locks) live in
# tests/unit/ with the amr-doctest-port plan -- their targets are declared
# in tests/unit/CMakeLists.txt but emit into build/tests/ so the run
# launchers (tests/unit/test_amr_units.py, the pytest wrappers, the
# bit-identity harness) keep their historical binary paths. The nesting
# mock sim lives in tests/regression/ (build/tests/regression/).
