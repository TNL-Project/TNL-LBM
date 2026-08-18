set(TARGETS
        sim_AMR
)

# optional C2F reconstruction strategy switch for AMR experiments (see
# include/lbm3d/d3q27/amr_coupling.h); e.g. configure with
#   cmake -B build -DTNL_LBM_C2F_STRATEGY=C2F_LINEAR_EXPLOSION
# The default (empty) builds the 3rd-order Lagrange interpolation.
set(TNL_LBM_C2F_STRATEGY "" CACHE STRING "C2F strategy define for AMR targets (C2F_LINEAR_EXPLOSION, C2F_UNIFORM_EXPLOSION, C2F_TRILINEAR, C2F_COMPACT_MOMENT)")

# optional F2C filter strategy switch for AMR experiments (see
# include/lbm3d/d3q27/amr_coupling.h); e.g. configure with
#   cmake -B build -DTNL_LBM_F2C_STRATEGY=F2C_BOX_AVERAGE
# The default (empty) builds the 4x4x4 Lagrange filter; F2C_BOX_AVERAGE
# opts out to the original 1/8 box average of the 8 subcells.
set(TNL_LBM_F2C_STRATEGY "" CACHE STRING "F2C filter strategy define for AMR targets (F2C_BOX_AVERAGE)")

# optional C2F carve switch for AMR experiments (change 5 of the AMR
# interface redesign; active only in the C2F_COMPACT_MOMENT branch -- the
# 2-cell window shifts one cell outward wherever it would touch a covered
# GEO_NOTHING cell); e.g. configure with
#   cmake -B build-a2 -DTNL_LBM_C2F_STRATEGY=C2F_COMPACT_MOMENT -DC2F_CARVE=ON
set(C2F_CARVE "" CACHE STRING "C2F carve define for AMR targets (requires TNL_LBM_C2F_STRATEGY=C2F_COMPACT_MOMENT; enable with ON)")

# 2026-08-16 (gate B ruling + D.1): the F2C skin feedback became the ONLY
# fine-to-coarse channel -- the former F2C_SKIN_ONLY experiment define was
# retired together with the ring-F2C code it gated, so there is no feedback
# switch left to configure.

foreach(target IN ITEMS ${TARGETS})
    add_executable(${target} ${target}.cu)
    target_link_libraries(${target} PUBLIC TNL::TNL_LBM)

    if(TNL_LBM_C2F_STRATEGY)
        target_compile_definitions(${target} PRIVATE ${TNL_LBM_C2F_STRATEGY})
    endif()

    if(TNL_LBM_F2C_STRATEGY)
        target_compile_definitions(${target} PRIVATE ${TNL_LBM_F2C_STRATEGY})
    endif()

    if(C2F_CARVE)
        target_compile_definitions(${target} PRIVATE C2F_CARVE)
    endif()

    # force language to HIP
    if(TNL_LBM_BUILD_HIP)
        set_source_files_properties(${target}.cu PROPERTIES LANGUAGE HIP)
    endif()
endforeach()
