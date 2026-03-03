set(TARGETS
        sim_adjoint
        sim_pseudomeasure
)

foreach(target IN ITEMS ${TARGETS})
    add_executable(${target} ${target}.cu)
    target_link_libraries(${target} PUBLIC TNL::TNL_LBM)

    # force language to HIP
    if(CMAKE_HIP_COMPILER)
        set_source_files_properties(${target}.cu PROPERTIES LANGUAGE HIP)
    endif()
endforeach()
