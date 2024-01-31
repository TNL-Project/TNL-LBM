set(TARGETS
        sim_1
        sim_2
        sim_3
        sim_4
)

foreach(target IN ITEMS ${TARGETS})
    # create the target
    add_executable(${target} ${target}.cu)

    # force language to HIP
    if(CMAKE_HIP_COMPILER)
        set_source_files_properties(${target}.cu PROPERTIES LANGUAGE HIP)
    endif()

    # add special flags used in lbm3d
    target_compile_definitions(${target} PUBLIC "-DUSE_CUDA")

    # add dependencies to the target
    target_link_libraries(${target} PUBLIC TNL::TNL_CUDA PNG::PNG fmt::fmt spdlog::spdlog)
    if(CMAKE_HIP_COMPILER)
        target_link_libraries(${target} PUBLIC TNL::TNL_HIP)
    endif()

    # enable OpenMP for the target
    if(OPENMP_FOUND)
        target_compile_definitions(${target} PUBLIC "-DHAVE_OPENMP")
        target_link_libraries(${target} PUBLIC OpenMP::OpenMP_CXX)
        # nvcc needs -Xcompiler
        target_compile_options(${target} PUBLIC $<$<CUDA_COMPILER_ID:NVIDIA>: -Xcompiler=-fopenmp >)
    endif()

    # enable MPI for the target
    if(MPI_FOUND)
        target_compile_definitions(${target} PUBLIC "-DHAVE_MPI")
        target_link_libraries(${target} PUBLIC MPI::MPI_CXX)
    endif()
endforeach()
