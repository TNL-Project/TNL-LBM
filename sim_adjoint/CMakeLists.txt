set(TARGETS
        sim_pseudomeasure
)

# sim_adjoint requires the A-B streaming pattern: the adjoint reversed gather is racy in a single DF array -- see AGENTS.md
if(NOT TNL_LBM_AA_PATTERN)
    list(APPEND TARGETS sim_adjoint)
endif()

foreach(target IN ITEMS ${TARGETS})
    add_executable(${target} ${target}.cu)
    target_link_libraries(${target} PUBLIC TNL::TNL_LBM)

    # force language to HIP
    if(TNL_LBM_BUILD_HIP)
        set_source_files_properties(${target}.cu PROPERTIES LANGUAGE HIP)
    endif()
endforeach()
