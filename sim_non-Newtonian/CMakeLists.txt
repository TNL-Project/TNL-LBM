set(TARGETS
        sim_nonnewtonian_poiseuille
        sim_nonnewtonian_poiseuille_CY
        sim_nonnewtonian_poiseuille_Casson
)

foreach(target IN ITEMS ${TARGETS})
    add_executable(${target} ${target}.cu)
    target_link_libraries(${target} PUBLIC TNL::TNL_LBM)

    # force language to HIP
    if(TNL_LBM_BUILD_HIP)
        set_source_files_properties(${target}.cu PROPERTIES LANGUAGE HIP)
    endif()
endforeach()
