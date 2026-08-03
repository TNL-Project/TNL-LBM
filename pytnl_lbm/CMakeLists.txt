set(sources
    pytnl_lbm.cpp
)

foreach(source_file IN LISTS sources)
    if(TNL_LBM_BUILD_HIP)
        set_source_files_properties(${source_file} PROPERTIES LANGUAGE HIP)
    elseif(TNL_LBM_BUILD_CUDA)
        set_source_files_properties(${source_file} PROPERTIES LANGUAGE CUDA)
    endif()
endforeach(source_file)

# Create the module
nanobind_add_module(pytnl_lbm ${sources})

# Add TNL-LBM and PyTNL
target_link_libraries(pytnl_lbm PUBLIC TNL::TNL_LBM PyTNL::PyTNL)

# Generate the typing stub for static type checking (same pattern as PyTNL).
# PYTHONPATH makes stubgen import the freshly built module from the build dir.
# pytnl_lbm imports the `pytnl` python package at init; when PyTNL comes from
# FetchContent (no system install, e.g. CI), that package only exists under
# ${pytnl_BINARY_DIR}/src and must be on PYTHONPATH as well.
# OMPI_MCA_accelerator=null avoids MPI_Init aborts on builders where Open MPI
# detects unsupported GPU accelerators (e.g. both ROCm and CUDA present).
find_package(Python 3 COMPONENTS Interpreter REQUIRED)
if(EXISTS ${NB_DIR}/stubgen.py)
    set(NB_STUBGEN ${NB_DIR}/stubgen.py)
elseif(EXISTS ${NB_DIR}/src/stubgen.py)
    set(NB_STUBGEN ${NB_DIR}/src/stubgen.py)
else()
    message(FATAL_ERROR "pytnl_lbm: could not find stubgen.py under NB_DIR=${NB_DIR}")
endif()
# pytnl_BINARY_DIR is set by FetchContent only when PyTNL is fetched, not when
# it is found as a system package (where the pytnl package is already installed)
if(pytnl_BINARY_DIR)
    set(pytnl_stub_pythonpath ":${pytnl_BINARY_DIR}/src")
    set(pytnl_stub_inputs -i "${pytnl_BINARY_DIR}/src")
else()
    set(pytnl_stub_pythonpath "")
    set(pytnl_stub_inputs "")
endif()
set(pytnl_lbm_stub "${CMAKE_CURRENT_BINARY_DIR}/pytnl_lbm.pyi")
add_custom_command(
    OUTPUT "${pytnl_lbm_stub}"
    COMMAND
        ${CMAKE_COMMAND} -E env PYTHONPATH=$<TARGET_FILE_DIR:pytnl_lbm>${pytnl_stub_pythonpath}
        OMPI_MCA_accelerator=null ${Python_EXECUTABLE} ${NB_STUBGEN}
        -q -r -m pytnl_lbm
        -i $<TARGET_FILE_DIR:pytnl_lbm> -i $<TARGET_FILE_DIR:pytnl_lbm>/..
        ${pytnl_stub_inputs}
        -O ${CMAKE_CURRENT_BINARY_DIR}
    WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
    DEPENDS pytnl_lbm "${NB_STUBGEN}"
)
add_custom_target(pytnl_lbm_stub ALL DEPENDS "${pytnl_lbm_stub}")

# Set installation path
install(TARGETS pytnl_lbm DESTINATION ${PyTNL_PYTHON_SITE_PACKAGES_DIR})
install(FILES "${pytnl_lbm_stub}" DESTINATION ${PyTNL_PYTHON_SITE_PACKAGES_DIR})

# Add include directory for mpi4py
if(MPI_FOUND AND TNL_LBM_USE_MPI)
    find_package(Python 3 COMPONENTS Interpreter REQUIRED)
    execute_process(
        COMMAND "${Python_EXECUTABLE}" -m mpi4py --prefix
        OUTPUT_STRIP_TRAILING_WHITESPACE
        OUTPUT_VARIABLE mpi4py_ROOT
    )
    target_include_directories(pytnl_lbm SYSTEM PRIVATE ${mpi4py_ROOT}/include)
endif()
