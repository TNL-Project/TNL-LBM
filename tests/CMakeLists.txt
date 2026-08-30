add_subdirectory(integration)
add_subdirectory(unit)

# The AMR doctest suites (coupling/subcycling/vtkhdf_writer/nesting, the C2F
# debug-define smokes, and the F2C Schönherr exactness locks) moved to
# tests/unit/ with the amr-doctest-port plan -- their targets are declared
# in tests/unit/CMakeLists.txt but emit into build/tests/ so the run
# scripts (run-amr-tests.sh, the pytest wrappers, the bit-identity harness)
# keep their historical binary paths.

# Dedicated 3-level nesting mock simulation for the ParaView end-to-end
# nesting arm (tests/test_amr_paraview_e2e_nesting.sh, the amr-nlevel-nesting
# plan's commit D): sim_AMR/sim_AMR_channel hardcode their region specs, so
# a compact driver tests/test_amr_nesting_sim.cu produces the nested-region
# VTKHDF frames the e2e script asserts. Single-pattern build (the e2e pair
# drives the default AB build), run by tests/run-amr-tests.sh as target #10
add_executable(test_amr_nesting_sim test_amr_nesting_sim.cu)
target_link_libraries(test_amr_nesting_sim PUBLIC TNL::TNL_LBM)

# force language to HIP
if(TNL_LBM_BUILD_HIP)
    set_source_files_properties(test_amr_nesting_sim.cu PROPERTIES LANGUAGE HIP)
endif()
