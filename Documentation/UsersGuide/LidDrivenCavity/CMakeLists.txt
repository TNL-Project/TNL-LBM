set(LIDCAVITY_EXAMPLES
    lid_driven_cavity_srt
)

foreach(target IN LISTS LIDCAVITY_EXAMPLES)
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
    list(APPEND LIDCAVITY_OUTPUTS ${TNL_LBM_DOCUMENTATION_OUTPUT_SNIPPETS_PATH}/${target}.out)
endforeach()

if(LIDCAVITY_OUTPUTS)
    add_custom_target(RunUGLidDrivenCavityExamples ALL DEPENDS ${LIDCAVITY_OUTPUTS})
    add_dependencies(${TNL_LBM_RUN_DOC_TARGET} RunUGLidDrivenCavityExamples)
endif()

configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/ug_LidDrivenCavity.md
    ${CMAKE_CURRENT_BINARY_DIR}/ug_LidDrivenCavity.md
    COPYONLY
)
