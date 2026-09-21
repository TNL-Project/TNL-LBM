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

# per-pattern test binaries: the pattern is pinned per target via the
# umbrella TNL_LBM_STREAMING_PATTERN_{AB_PULL,AA} definitions consumed by
# the streaming.h alias tables. The TNL_LBM interface target always
# propagates the tree-wide selection, but the alias tables match
# TNL_LBM_STREAMING_PATTERN_AA first, so the aa pin wins in a default (A-B)
# tree; in an AA-configured tree the same precedence would silently turn
# the "ab" binaries into second AA builds, so they are created only in the
# default A-B tree (the AA tree still gets the aa-pinned binaries and the
# default-pattern test_cpp_units)
if(TNL_LBM_STREAMING_PATTERN STREQUAL "AA")
    set(AMR_TEST_PATTERNS aa)
    message(STATUS "tests/unit: AA-pattern build tree - skipping the AB-pinned per-pattern test targets")
else()
    set(AMR_TEST_PATTERNS ab aa)
endif()

# force language to HIP
if(TNL_LBM_BUILD_HIP)
    set_source_files_properties(${unit_test_sources} PROPERTIES LANGUAGE HIP)
endif()

# doctest-based AMR gate binaries (the amr-doctest-port plan): the four mock
# suites consolidated per streaming pattern -- coupling, subcycling,
# vtkhdf_writer and nesting, each a TEST_SUITE inside one binary. Emitted
# into build/tests/ so tests/unit/test_amr_units.py, the bit-identity
# harness and the pytest wrappers keep their historical binary paths.
set(AMR_SUITE_SOURCES
    test_amr_coupling.cu
    test_amr_subcycling.cu
    test_amr_vtkhdf_writer.cu
    test_amr_nesting.cu
)

# the "ab"/"aa" binary suffixes pin the A-B pull reference and the A-A
# pattern respectively via the umbrella selection definitions; the pin is
# applied through COMPILE OPTIONS so it lands after the DEFINES group on
# the compile command line (strip the tree-level interface define, then
# pin): the umbrella's #if/#elif first-match precedence otherwise defeats
# a plain compile definition whenever the tree pattern is checked earlier
# than the pinned one (e.g. an AB_PUSH tree vs the AB_PULL pin)
function(amr_pattern_define pattern out_var)
    if(pattern STREQUAL "aa")
        set(pin_define "TNL_LBM_STREAMING_PATTERN_AA")
    else()
        set(pin_define "TNL_LBM_STREAMING_PATTERN_AB_PULL")
    endif()
    set(opts "-D${pin_define}")
    if(TNL_LBM_STREAMING_PATTERN)
        list(PREPEND opts "-UTNL_LBM_STREAMING_PATTERN_${TNL_LBM_STREAMING_PATTERN}")
    endif()
    set(${out_var} "${opts}" PARENT_SCOPE)
endfunction()

foreach(pattern IN ITEMS ${AMR_TEST_PATTERNS})
    amr_pattern_define(${pattern} pattern_options)
    set(target "test_amr_units_${pattern}")
    add_executable(${target} doctest_main_amr.cu ${AMR_SUITE_SOURCES})
    target_compile_options(${target} PRIVATE ${pattern_options})
    target_link_libraries(${target} PUBLIC TNL::TNL_LBM PRIVATE doctest::doctest)
    set_target_properties(${target} PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/tests")

    # optional C2F/F2C strategy compile-checks (same cache vars as sim_AMR):
    # the suites pin BOTH strategies natively via compile-time
    # strategy-split expectations and the source-level deferral arms, so
    # these binaries compile and run green under either strategy build;
    # the pytest gate module itself drives only the default build
    if(TNL_LBM_C2F_STRATEGY)
        target_compile_definitions(${target} PRIVATE ${TNL_LBM_C2F_STRATEGY})
    endif()
    if(TNL_LBM_F2C_STRATEGY)
        target_compile_definitions(${target} PRIVATE ${TNL_LBM_F2C_STRATEGY})
    endif()

    # force language to HIP
    if(TNL_LBM_BUILD_HIP)
        set_source_files_properties(doctest_main_amr.cu ${AMR_SUITE_SOURCES} PROPERTIES LANGUAGE HIP)
    endif()
endforeach()

# standalone per-define doctest driver: the F2C_SCHONHERR branch selects a
# different body of the same kernel template symbol (ODR hazard), so the
# define gets its own binary, built from doctest_main.cu + the shared driver
# TU. Driven by tests/unit/test_amr_f2c_schonherr.py, NOT by the gate module
foreach(pattern IN ITEMS ${AMR_TEST_PATTERNS})
    amr_pattern_define(${pattern} pattern_options)
    set(target "test_amr_f2c_schonherr_${pattern}")
    add_executable(${target} doctest_main_amr.cu test_amr_f2c_schonherr.cu)
    target_compile_definitions(${target} PRIVATE "F2C_SCHONHERR")
    target_compile_options(${target} PRIVATE ${pattern_options})
    target_link_libraries(${target} PUBLIC TNL::TNL_LBM PRIVATE doctest::doctest)
    set_target_properties(${target} PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/tests")

    # force language to HIP
    if(TNL_LBM_BUILD_HIP)
        set_source_files_properties(doctest_main_amr.cu test_amr_f2c_schonherr.cu PROPERTIES LANGUAGE HIP)
    endif()
endforeach()

# an AA-configured tree does not build the AB-pinned targets above: drop their
# stale outputs left over from a previous A-B configure, so that test drivers
# locating the binaries by path (e.g. tests/unit/test_amr_units.py) cannot
# pick up outdated AB binaries that this tree no longer builds (a missing
# binary is an honest hard failure; a stale one is silently wrong)
if(TNL_LBM_STREAMING_PATTERN STREQUAL "AA")
    file(REMOVE
        ${CMAKE_BINARY_DIR}/tests/test_amr_units_ab
        ${CMAKE_BINARY_DIR}/tests/test_amr_f2c_schonherr_ab
    )
endif()
