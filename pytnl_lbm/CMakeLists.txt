set(sources
    pytnl_lbm.cpp
)

foreach(source_file IN LISTS sources)
    set_source_files_properties(${source_file} PROPERTIES LANGUAGE CUDA)
endforeach(source_file)

# Create the module
nanobind_add_module(pytnl_lbm ${sources})

# Add TNL-LBM and PyTNL
target_link_libraries(pytnl_lbm PUBLIC TNL::TNL_LBM PyTNL::PyTNL)

# Set installation path
install(TARGETS pytnl_lbm DESTINATION ${PyTNL_PYTHON_SITE_PACKAGES_DIR})

# Add include directory for mpi4py
if(MPI_FOUND AND TNL_LBM_ENABLE_MPI)
    find_package(Python 3 COMPONENTS Interpreter REQUIRED)
    execute_process(
        COMMAND "${Python_EXECUTABLE}" -m mpi4py --prefix
        OUTPUT_STRIP_TRAILING_WHITESPACE
        OUTPUT_VARIABLE mpi4py_ROOT
    )
    target_include_directories(pytnl_lbm SYSTEM PRIVATE ${mpi4py_ROOT}/include)
endif()
