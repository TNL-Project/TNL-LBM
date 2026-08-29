# one doctest binary for all unit-test .cu files in this directory
# (doctest_main.cu carries the runner main function, the rest are test cases)
set(unit_test_sources
    test_outflowcover.cu
    test_outflow_gather2d.cu
    test_outflow_gather3d.cu
    test_decomposition.cu
    test_bc_symmetry.cu
    doctest_main.cu
)

add_executable(test_cpp_units ${unit_test_sources})
target_link_libraries(test_cpp_units PUBLIC TNL::TNL_LBM PRIVATE doctest::doctest)

# force language to HIP
if(TNL_LBM_BUILD_HIP)
    set_source_files_properties(${unit_test_sources} PROPERTIES LANGUAGE HIP)
endif()
