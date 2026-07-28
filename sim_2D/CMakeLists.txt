set(TARGETS
    sim2d_1
    sim2d_2
    sim2d_Taylor_Green
)

foreach(target IN ITEMS ${TARGETS})
    add_executable(${target} ${target}.cu)
    target_link_libraries(${target} PUBLIC TNL::TNL_LBM)

    # force language to HIP
    if(CMAKE_HIP_COMPILER)
        set_source_files_properties(${target}.cu PROPERTIES LANGUAGE HIP)
    endif()
endforeach()
