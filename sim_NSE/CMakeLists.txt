set(TARGETS
        sim_1
        sim_2
        sim_3
        sim_4
        sim_IBM1
        sim_IBM2
        sim_IBM3
        sim_IBM4
)

foreach(target IN ITEMS ${TARGETS})
    add_executable(${target} ${target}.cu)
    target_link_libraries(${target} PUBLIC TNL::TNL_LBM)

    # force language to HIP
    if(CMAKE_HIP_COMPILER)
        set_source_files_properties(${target}.cu PROPERTIES LANGUAGE HIP)
    endif()
endforeach()
