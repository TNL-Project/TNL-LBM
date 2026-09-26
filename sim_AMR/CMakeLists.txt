set(TARGETS
        sim_AMR
        sim_AMR_channel
        sim_AMR_ball
)

# optional C2F reconstruction strategy switch for AMR experiments (see
# include/lbm3d/d3q27/amr_coupling.h); e.g. configure with
#   cmake -B build -DTNL_LBM_C2F_STRATEGY=C2F_LAGRANGE
# The default (empty) builds the compact-moment interpolation; before the
# flip the default was the 3rd-order Lagrange interpolation (now the
# C2F_LAGRANGE opt-out).
set(TNL_LBM_C2F_STRATEGY "" CACHE STRING "C2F strategy define for AMR targets (C2F_LAGRANGE, C2F_TRILINEAR, C2F_LINEAR_EXPLOSION, C2F_UNIFORM_EXPLOSION; C2F_COMPACT_MOMENT is the default, accepted as a no-op)")

# optional F2C transfer strategy switch for AMR experiments (see
# include/lbm3d/d3q27/amr_coupling.h); e.g. configure with
#   cmake -B build -DTNL_LBM_F2C_STRATEGY=F2C_LAGRAVA
# The default (empty) builds the thesis sec. 7.2 sigma-form compact-moment
# transfer (sigma = 2; no tau-rescale, no f_avg, no lo = 0 clamp, no
# 4x4x4 window) -- the F2C_SCHONHERR strategy, the default since
# 2026-08-22 with the Schönherr ch7 conversion, untagged
# since 2026-09-27 when the branch split flipped from
# #ifdef F2C_SCHONHERR to the opt-out-only form; the retired
# F2C_SCHONHERR value itself is accepted as a no-op, so pre-flip build
# caches self-heal onto the default branch.
# -DTNL_LBM_F2C_STRATEGY=F2C_LAGRAVA opts out to the 4x4x4 Lagrange
# filter (the pre-conversion default); F2C_BOX_AVERAGE selects the
# original 1/8 box average of the 8 subcells inside that opt-out branch.
set(TNL_LBM_F2C_STRATEGY "" CACHE STRING "F2C transfer strategy define for AMR targets (F2C_SCHONHERR is the default, accepted as a no-op; F2C_LAGRAVA opts out to the 4x4x4 Lagrange filter, F2C_BOX_AVERAGE to the 1/8 box average)")

# 2026-08-16: the F2C skin feedback became the ONLY
# fine-to-coarse channel -- the former F2C_SKIN_ONLY experiment define was
# retired together with the ring-F2C code it gated, so there is no feedback
# switch left to configure.

# built only in MPI-configured trees: the AMR driver is a single-MPI-rank
# design and the sims include mpi.h directly (a non-MPI tree cannot build
# them -- TNL's DummyDefs.h stub collides with the true header)
if(MPI_FOUND AND TNL_LBM_USE_MPI)
foreach(target IN ITEMS ${TARGETS})
    add_executable(${target} ${target}.cu)
    target_link_libraries(${target} PUBLIC TNL::TNL_LBM)

    if(TNL_LBM_C2F_STRATEGY)
        target_compile_definitions(${target} PRIVATE ${TNL_LBM_C2F_STRATEGY})
    endif()

    if(TNL_LBM_F2C_STRATEGY)
        target_compile_definitions(${target} PRIVATE ${TNL_LBM_F2C_STRATEGY})
    endif()

    # force language to HIP
    if(TNL_LBM_BUILD_HIP)
        set_source_files_properties(${target}.cu PROPERTIES LANGUAGE HIP)
    endif()
endforeach()
endif()
