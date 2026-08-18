set(TARGETS
        sim_AMR
        sim_AMR_channel
)

# optional C2F reconstruction strategy switch for AMR experiments (see
# include/lbm3d/d3q27/amr_coupling.h); e.g. configure with
#   cmake -B build -DTNL_LBM_C2F_STRATEGY=C2F_LAGRANGE
# 2026-08-18 (user ruling): the default (empty) builds the compact-moment
# interpolation with the carve pre-pass -- the interpolation never reads
# covered GEO_NOTHING cells; before the flip the default was the 3rd-order
# Lagrange interpolation (now the C2F_LAGRANGE opt-out).
set(TNL_LBM_C2F_STRATEGY "" CACHE STRING "C2F strategy define for AMR targets (C2F_LAGRANGE, C2F_TRILINEAR, C2F_LINEAR_EXPLOSION, C2F_UNIFORM_EXPLOSION; C2F_COMPACT_MOMENT is the default, accepted as a no-op)")

# optional F2C filter strategy switch for AMR experiments (see
# include/lbm3d/d3q27/amr_coupling.h); e.g. configure with
#   cmake -B build -DTNL_LBM_F2C_STRATEGY=F2C_BOX_AVERAGE
# The default (empty) builds the 4x4x4 Lagrange filter; F2C_BOX_AVERAGE
# opts out to the original 1/8 box average of the 8 subcells.
set(TNL_LBM_F2C_STRATEGY "" CACHE STRING "F2C filter strategy define for AMR targets (F2C_BOX_AVERAGE)")

# optional C2F carve opt-out switch (change 5 of the AMR interface
# redesign -- the 2-cell CM window shifts one cell outward wherever it
# would touch a covered GEO_NOTHING cell). 2026-08-18 (user ruling): carve
# is default-ON; opt out with -DC2F_NO_CARVE=ON. The former C2F_CARVE
# opt-in is retired and ignored (defining it is a no-op).
set(C2F_NO_CARVE "" CACHE STRING "C2F carve opt-out define for AMR targets (carve is default-on; set ON to disable)")
if(C2F_CARVE)
    message(WARNING "C2F_CARVE was retired on 2026-08-18: the carve is default-on and the define is a no-op; use C2F_NO_CARVE=ON to opt out.")
endif()

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

    if(C2F_NO_CARVE)
        target_compile_definitions(${target} PRIVATE C2F_NO_CARVE)
    endif()

    # force language to HIP
    if(TNL_LBM_BUILD_HIP)
        set_source_files_properties(${target}.cu PROPERTIES LANGUAGE HIP)
    endif()
endforeach()
