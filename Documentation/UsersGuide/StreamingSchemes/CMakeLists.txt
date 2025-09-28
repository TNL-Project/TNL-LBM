set(STREAMING_EXAMPLES
    lid_driven_cavity_aa
)

foreach(target IN LISTS STREAMING_EXAMPLES)
    add_executable(${target} ${target}.cu)
    target_link_libraries(${target} PUBLIC TNL::TNL_LBM)
    target_compile_definitions(${target} PRIVATE DOXYGEN_DOCUMENTATION=1)
    target_compile_definitions(${target} PRIVATE AA_PATTERN)
    add_custom_command(
        COMMAND ${target} --doc-mode > ${TNL_LBM_DOCUMENTATION_OUTPUT_SNIPPETS_PATH}/${target}.out
        OUTPUT ${TNL_LBM_DOCUMENTATION_OUTPUT_SNIPPETS_PATH}/${target}.out
        DEPENDS ${target}
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
        VERBATIM
    )
    list(APPEND STREAMING_OUTPUTS ${TNL_LBM_DOCUMENTATION_OUTPUT_SNIPPETS_PATH}/${target}.out)
endforeach()

if(STREAMING_OUTPUTS)
    add_custom_target(RunUGStreamingExamples ALL DEPENDS ${STREAMING_OUTPUTS})
    add_dependencies(${TNL_LBM_RUN_DOC_TARGET} RunUGStreamingExamples)
endif()

configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/ug_StreamingSchemes.md
    ${CMAKE_CURRENT_BINARY_DIR}/ug_StreamingSchemes.md
    COPYONLY
)
