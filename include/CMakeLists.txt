install( DIRECTORY lbm_common lbm2d lbm3d
         TYPE INCLUDE
         FILES_MATCHING PATTERN "*.h" PATTERN "*.hpp" )

# clangd support: compile_commands.json only lists translation units,
# so clangd would infer a bogus compile command for every library header
# (wrong include paths, wrong C++ standard).
# Emit a real entry per header via an OBJECT library that is never built.
if(CMAKE_EXPORT_COMPILE_COMMANDS)
    file(GLOB_RECURSE tnl_lbm_clangd_headers CONFIGURE_DEPENDS "lbm_common/*.h" "lbm3d/*.h" "lbm3d/*.hpp")
    # the nanobind wrapper headers (py_*.h) get their own entries from
    # pytnl_lbm/CMakeLists.txt with the richer include set of the module
    list(FILTER tnl_lbm_clangd_headers EXCLUDE REGEX "/py_[^/]*\\.h$")
    add_library(tnl_lbm_clangd_headers OBJECT EXCLUDE_FROM_ALL ${tnl_lbm_clangd_headers})
    # compile the entries with the same toolchain and flags as the real code
    if(CMAKE_CUDA_COMPILER)
        set_source_files_properties(${tnl_lbm_clangd_headers} PROPERTIES LANGUAGE CUDA)
    elseif(CMAKE_HIP_COMPILER)
        set_source_files_properties(${tnl_lbm_clangd_headers} PROPERTIES LANGUAGE HIP)
    else()
        set_source_files_properties(${tnl_lbm_clangd_headers} PROPERTIES LANGUAGE CXX)
    endif()
    target_link_libraries(tnl_lbm_clangd_headers PRIVATE TNL_LBM)
endif()
