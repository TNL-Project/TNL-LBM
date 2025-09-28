set(COLLISION_EXAMPLES
    lid_driven_cavity_mrt
    lid_driven_cavity_clbm
)

foreach(target IN LISTS COLLISION_EXAMPLES)
    add_executable(${target} ${target}.cu)
    target_link_libraries(${target} PUBLIC TNL::TNL_LBM)
    target_compile_definitions(${target} PRIVATE DOXYGEN_DOCUMENTATION=1)
    add_custom_command(
        COMMAND ${target} --doc-mode > ${TNL_LBM_DOCUMENTATION_OUTPUT_SNIPPETS_PATH}/${target}.out
        OUTPUT ${TNL_LBM_DOCUMENTATION_OUTPUT_SNIPPETS_PATH}/${target}.out
        DEPENDS ${target}
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
        VERBATIM
    )
    list(APPEND COLLISION_OUTPUTS ${TNL_LBM_DOCUMENTATION_OUTPUT_SNIPPETS_PATH}/${target}.out)
endforeach()

if(COLLISION_OUTPUTS)
    add_custom_target(RunUGCollisionOperatorExamples ALL DEPENDS ${COLLISION_OUTPUTS})
    add_dependencies(${TNL_LBM_RUN_DOC_TARGET} RunUGCollisionOperatorExamples)
endif()

configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/ug_CollisionOperators.md
    ${CMAKE_CURRENT_BINARY_DIR}/ug_CollisionOperators.md
    COPYONLY
)
