add_subdirectory(integration)
add_subdirectory(unit)

# AMR coupling and subcycling unit tests (run by tests/run-amr-tests.sh):
# one source per test suite, compiled once per streaming pattern
set(AMR_TESTS
        test_amr_coupling
        test_amr_subcycling
)

foreach(test_name IN ITEMS ${AMR_TESTS})
    foreach(pattern IN ITEMS ab aa)
        string(TOUPPER "${pattern}" pattern_upper)
        set(target "${test_name}_${pattern}")
        add_executable(${target} ${test_name}.cu)
        target_compile_definitions(${target} PRIVATE "${pattern_upper}_PATTERN")
        target_link_libraries(${target} PUBLIC TNL::TNL_LBM)

        # force language to HIP
        if(TNL_LBM_BUILD_HIP)
            set_source_files_properties(${test_name}.cu PROPERTIES LANGUAGE HIP)
        endif()
    endforeach()
endforeach()
