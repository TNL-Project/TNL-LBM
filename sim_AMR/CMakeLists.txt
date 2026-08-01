set(TARGETS
        sim_AMR
)

# optional C2F reconstruction strategy switch for AMR experiments (see
# include/lbm3d/d3q27/amr_coupling.h); e.g. configure with
#   cmake -B build -DTNL_LBM_C2F_STRATEGY=C2F_LINEAR_EXPLOSION
# The default (empty) builds the 3rd-order Lagrange interpolation.
set(TNL_LBM_C2F_STRATEGY "" CACHE STRING "C2F strategy define for AMR targets (C2F_LINEAR_EXPLOSION, C2F_UNIFORM_EXPLOSION, C2F_TRILINEAR)")

foreach(target IN ITEMS ${TARGETS})
    add_executable(${target} ${target}.cu)
    target_link_libraries(${target} PUBLIC TNL::TNL_LBM)

    if(TNL_LBM_C2F_STRATEGY)
        target_compile_definitions(${target} PRIVATE ${TNL_LBM_C2F_STRATEGY})
    endif()

    # force language to HIP
    if(TNL_LBM_BUILD_HIP)
        set_source_files_properties(${target}.cu PROPERTIES LANGUAGE HIP)
    endif()
endforeach()
