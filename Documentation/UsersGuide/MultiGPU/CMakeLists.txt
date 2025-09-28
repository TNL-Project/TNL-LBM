set(MULTIGPU_EXAMPLES
    lid_driven_cavity_multi_gpu
)

foreach(target IN LISTS MULTIGPU_EXAMPLES)
    add_executable(${target} ${target}.cu)
    target_link_libraries(${target} PUBLIC TNL::TNL_LBM)
    target_compile_definitions(${target} PRIVATE DOXYGEN_DOCUMENTATION=1)
    if(MPIEXEC_EXECUTABLE)
        add_custom_command(
            COMMAND ${MPIEXEC_EXECUTABLE} -n 1 $<TARGET_FILE:${target}> --doc-mode > ${TNL_LBM_DOCUMENTATION_OUTPUT_SNIPPETS_PATH}/${target}.out
            OUTPUT ${TNL_LBM_DOCUMENTATION_OUTPUT_SNIPPETS_PATH}/${target}.out
            DEPENDS ${target}
            WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
            VERBATIM
        )
        list(APPEND MULTIGPU_OUTPUTS ${TNL_LBM_DOCUMENTATION_OUTPUT_SNIPPETS_PATH}/${target}.out)
    endif()
endforeach()

if(MULTIGPU_OUTPUTS)
    add_custom_target(RunUGMultiGPUExamples ALL DEPENDS ${MULTIGPU_OUTPUTS})
    add_dependencies(${TNL_LBM_RUN_DOC_TARGET} RunUGMultiGPUExamples)
endif()

configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/ug_MultiGPU.md
    ${CMAKE_CURRENT_BINARY_DIR}/ug_MultiGPU.md
    COPYONLY
)
