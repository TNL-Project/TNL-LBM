add_subdirectory(integration)
add_subdirectory(unit)

# AMR coupling and subcycling unit tests (run by tests/run-amr-tests.sh):
# one source per test suite, compiled once per streaming pattern
set(AMR_TESTS
        test_amr_coupling
        test_amr_subcycling
        test_amr_vtkhdf_writer
)

foreach(test_name IN ITEMS ${AMR_TESTS})
    foreach(pattern IN ITEMS ab aa)
        string(TOUPPER "${pattern}" pattern_upper)
        set(target "${test_name}_${pattern}")
        add_executable(${target} ${test_name}.cu)
        target_compile_definitions(${target} PRIVATE "${pattern_upper}_PATTERN")
        target_link_libraries(${target} PUBLIC TNL::TNL_LBM)

        # optional C2F strategy compile-check (same cache var as sim_AMR):
        # only for verifying that alternative strategies compile in both
        # patterns -- the tests pin the default scheme's semantics, so they
        # are RUN with the default build (empty TNL_LBM_C2F_STRATEGY)
        if(TNL_LBM_C2F_STRATEGY)
            target_compile_definitions(${target} PRIVATE ${TNL_LBM_C2F_STRATEGY})
        endif()

        # optional F2C filter strategy compile-check (same cache var as
        # sim_AMR): only for verifying that the opt-out compiles in both
        # patterns -- the tests pin the default filter's semantics, so they
        # are RUN with the default build (empty TNL_LBM_F2C_STRATEGY)
        if(TNL_LBM_F2C_STRATEGY)
            target_compile_definitions(${target} PRIVATE ${TNL_LBM_F2C_STRATEGY})
        endif()

        # optional C2F carve opt-out compile-check (same cache var as
        # sim_AMR): carve is default-on since the 2026-08-18 flip; only for
        # verifying that the C2F_NO_CARVE opt-out compiles in both patterns
        # -- the tests pin the default scheme's semantics, so they are RUN
        # with the default build
        if(C2F_NO_CARVE)
            target_compile_definitions(${target} PRIVATE C2F_NO_CARVE)
        endif()

        # 2026-08-16 (gate B ruling + D.1): the F2C skin feedback became
        # the ONLY fine-to-coarse channel -- F2C_SKIN_ONLY was retired with
        # the ring-F2C code, and the tests now always exercise the skin
        # path (lo = 0 window clamp unconditional)

        # force language to HIP
        if(TNL_LBM_BUILD_HIP)
            set_source_files_properties(${test_name}.cu PROPERTIES LANGUAGE HIP)
        endif()
    endforeach()
endforeach()

# Schönherr ch7 conversion (commit 10, plan T10g): compile-and-run smoke
# locks for the four C2F_*_ONLY seam-investigation debug defines of the C2F
# compact-moment branch. These are the compile-time -D switches that the
# interface-bias investigation rebuilds sim_AMR with (sim_AMR/CMakeLists.txt
# cache vars C2F_EQ_ONLY & co.); here they are HARDCODED per target (no new
# cache option is wired) so the smokes live in the default build. One binary
# per define per streaming pattern -- the same idiom as the AMR test
# binaries above. Driven by tests/unit/test_amr_c2f_debug_smoke.py, NOT by
# run-amr-tests.sh (the 7/7 suite stays untouched).
set(C2F_SMOKE_NAMES eq dev norm shear)
set(C2F_SMOKE_DEFINES C2F_EQ_ONLY C2F_DEV_ONLY C2F_NORM_ONLY C2F_SHEAR_ONLY)
foreach(smoke_name smoke_define IN ZIP_LISTS C2F_SMOKE_NAMES C2F_SMOKE_DEFINES)
    foreach(pattern IN ITEMS ab aa)
        string(TOUPPER "${pattern}" pattern_upper)
        set(target "test_amr_c2f_smoke_${smoke_name}_${pattern}")
        add_executable(${target} test_amr_c2f_debug_smoke.cu)
        target_compile_definitions(${target} PRIVATE "${pattern_upper}_PATTERN" "${smoke_define}")
        target_link_libraries(${target} PUBLIC TNL::TNL_LBM)

        # force language to HIP
        if(TNL_LBM_BUILD_HIP)
            set_source_files_properties(test_amr_c2f_debug_smoke.cu PROPERTIES LANGUAGE HIP)
        endif()
    endforeach()
endforeach()
