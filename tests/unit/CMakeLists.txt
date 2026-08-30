# one doctest binary for all unit-test .cu files in this directory
# (doctest_main.cu carries the runner main function, the rest are test cases)
set(unit_test_sources
    test_outflow_cover.cu
    test_outflow_gather2d.cu
    test_outflow_gather3d.cu
    test_streaming_patterns.cu
    test_inflow_moment.cu
    test_decomposition.cu
    test_bc_symmetry.cu
    test_streaming_mpi.cu
    test_streaming_mpi_bc2d.cu
    test_streaming_mpi_bc3d.cu
    test_amr_schonherr_registration.cu
    test_amr_schonherr_exactness.cu
    doctest_main.cu
)

add_executable(test_cpp_units ${unit_test_sources})
target_link_libraries(test_cpp_units PUBLIC TNL::TNL_LBM PRIVATE doctest::doctest)

# the streaming-MPI BC-dispatch cases (test_streaming_mpi_bc2d.cu /
# test_streaming_mpi_bc3d.cu) live in separate translation units per lattice
# model: the per-pattern BC switch instantiations dominate the compile time,
# and the split lets ninja schedule the model TUs in parallel

# force language to HIP
if(TNL_LBM_BUILD_HIP)
    set_source_files_properties(${unit_test_sources} PROPERTIES LANGUAGE HIP)
endif()

# doctest-based AMR gate binaries (the amr-doctest-port plan): the four mock
# suites consolidated per streaming pattern -- coupling, subcycling,
# vtkhdf_writer and nesting, each a TEST_SUITE inside one binary. Emitted
# into build/tests/ so run-amr-tests.sh, the bit-identity harness and the
# pytest wrappers keep their historical binary paths.
set(AMR_SUITE_SOURCES
    test_amr_coupling.cu
    test_amr_subcycling.cu
    test_amr_vtkhdf_writer.cu
    test_amr_nesting.cu
)

foreach(pattern IN ITEMS ab aa)
    string(TOUPPER "${pattern}" pattern_upper)
    set(target "test_amr_units_${pattern}")
    add_executable(${target} doctest_main.cu ${AMR_SUITE_SOURCES})
    target_compile_definitions(${target} PRIVATE "${pattern_upper}_PATTERN")
    target_link_libraries(${target} PUBLIC TNL::TNL_LBM PRIVATE doctest::doctest)
    set_target_properties(${target} PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/tests")

    # optional C2F/F2C strategy compile-checks (same cache vars as sim_AMR):
    # the suites pin BOTH strategies natively via compile-time
    # strategy-split expectations and the source-level deferral arms, so
    # these binaries compile and run green under either strategy build;
    # run-amr-tests.sh itself drives only the default build
    if(TNL_LBM_C2F_STRATEGY)
        target_compile_definitions(${target} PRIVATE ${TNL_LBM_C2F_STRATEGY})
    endif()
    if(TNL_LBM_F2C_STRATEGY)
        target_compile_definitions(${target} PRIVATE ${TNL_LBM_F2C_STRATEGY})
    endif()

    # force language to HIP
    if(TNL_LBM_BUILD_HIP)
        set_source_files_properties(${AMR_SUITE_SOURCES} PROPERTIES LANGUAGE HIP)
    endif()
endforeach()

# standalone per-define doctest driver: the F2C_SCHONHERR branch selects a
# different body of the same kernel template symbol (ODR hazard), so the
# define gets its own binary, built from doctest_main.cu + the shared driver
# TU. Driven by tests/unit/test_amr_f2c_schonherr.py, NOT by run-amr-tests.sh
foreach(pattern IN ITEMS ab aa)
    string(TOUPPER "${pattern}" pattern_upper)
    set(target "test_amr_f2c_schonherr_${pattern}")
    add_executable(${target} doctest_main.cu test_amr_f2c_schonherr.cu)
    target_compile_definitions(${target} PRIVATE "${pattern_upper}_PATTERN" "F2C_SCHONHERR")
    target_link_libraries(${target} PUBLIC TNL::TNL_LBM PRIVATE doctest::doctest)
    set_target_properties(${target} PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/tests")

    # force language to HIP
    if(TNL_LBM_BUILD_HIP)
        set_source_files_properties(test_amr_f2c_schonherr.cu PROPERTIES LANGUAGE HIP)
    endif()
endforeach()
