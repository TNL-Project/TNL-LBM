# Configure common output location for snippets
set(TNL_LBM_DOCUMENTATION_OUTPUT_SNIPPETS_PATH "${CMAKE_CURRENT_BINARY_DIR}/output_snippets")
file(MAKE_DIRECTORY ${TNL_LBM_DOCUMENTATION_OUTPUT_SNIPPETS_PATH})

# Aggregate target to run documentation examples (reuse upstream target if available)
if(TARGET run-doc-examples)
    set(TNL_LBM_RUN_DOC_TARGET run-doc-examples)
else()
    add_custom_target(run-doc-examples ALL)
    set(TNL_LBM_RUN_DOC_TARGET run-doc-examples)
endif()

# Users' guide with tutorials and examples
add_subdirectory(UsersGuide)

find_package(Doxygen)
if(DOXYGEN_FOUND)
    # Configure the Doxygen file and wire it to CMake
    set(TNL_LBM_DOXYGEN_INPUT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/UsersGuide")
    set(TNL_LBM_DOXYGEN_OUTPUT_DIR "${CMAKE_CURRENT_BINARY_DIR}")
    configure_file(
        ${CMAKE_CURRENT_SOURCE_DIR}/Doxyfile.in
        ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile
        @ONLY
    )

    add_custom_command(
        OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/doxygen.stamp
        COMMAND ${CMAKE_COMMAND} -E env
            PROJECT_NUMBER=${PROJECT_VERSION}
            OUTPUT_DIRECTORY=${TNL_LBM_DOXYGEN_OUTPUT_DIR}
            OUTPUT_SNIPPETS_PATH=${TNL_LBM_DOCUMENTATION_OUTPUT_SNIPPETS_PATH}
            ${DOXYGEN_EXECUTABLE} ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile
        COMMAND ${CMAKE_COMMAND} -E touch ${CMAKE_CURRENT_BINARY_DIR}/doxygen.stamp
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile
        COMMENT "Generating TNL-LBM Users' Guide with Doxygen"
        VERBATIM
    )

    add_custom_target(tnl-lbm-doxygen DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/doxygen.stamp)
    add_dependencies(tnl-lbm-doxygen ${TNL_LBM_RUN_DOC_TARGET})

    if(TARGET doxygen)
        add_dependencies(doxygen tnl-lbm-doxygen)
    else()
        add_custom_target(doxygen DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/doxygen.stamp)
        add_dependencies(doxygen ${TNL_LBM_RUN_DOC_TARGET})
    endif()
else()
    message(WARNING "Doxygen was not found, documentation targets will be skipped.")
endif()
