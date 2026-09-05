set(TARGETS
        sim_AMR
        sim_AMR_channel
        sim_AMR_ball
)

# optional C2F reconstruction strategy switch for AMR experiments (see
# include/lbm3d/d3q27/amr_coupling.h); e.g. configure with
#   cmake -B build -DTNL_LBM_C2F_STRATEGY=C2F_LAGRANGE
# 2026-08-18 (user ruling): the default (empty) builds the compact-moment
# interpolation with the carve pre-pass -- the interpolation never reads
# covered GEO_NOTHING cells; before the flip the default was the 3rd-order
# Lagrange interpolation (now the C2F_LAGRANGE opt-out).
set(TNL_LBM_C2F_STRATEGY "" CACHE STRING "C2F strategy define for AMR targets (C2F_LAGRANGE, C2F_TRILINEAR, C2F_LINEAR_EXPLOSION, C2F_UNIFORM_EXPLOSION; C2F_COMPACT_MOMENT is the default, accepted as a no-op)")

# optional F2C transfer strategy switch for AMR experiments (see
# include/lbm3d/d3q27/amr_coupling.h); e.g. configure with
#   cmake -B build -DTNL_LBM_F2C_STRATEGY=F2C_LAGRAVA
# The default (F2C_SCHONHERR, since 2026-08-22 -- commit 15 / T17 of the
# Schönherr ch7 conversion) builds the thesis sec. 7.2 sigma-form
# compact-moment transfer (sigma = 2; no tau-rescale, no f_avg, no
# lo = 0 clamp, no 4x4x4 window). -DTNL_LBM_F2C_STRATEGY=F2C_LAGRAVA
# opts out to the 4x4x4 Lagrange filter (the pre-conversion default): it
# is a named no-op define -- the kernel splits on #ifdef F2C_SCHONHERR
# only (see the #else branch of cudaAMR_FineToCoarse), so every other
# value, including the empty string, lands on the same Lagrava branch.
# F2C_BOX_AVERAGE keeps selecting the original 1/8 box average of the 8
# subcells inside that else-branch.
set(TNL_LBM_F2C_STRATEGY "F2C_SCHONHERR" CACHE STRING "F2C transfer strategy define for AMR targets (default F2C_SCHONHERR; F2C_LAGRAVA opts out to the 4x4x4 Lagrange filter, F2C_BOX_AVERAGE to the 1/8 box average)")

# 2026-08-23 (simulated-band flip): the C2F carve pre-pass was
# HARD-REMOVED -- under the ch7 band registration every nominal source
# window of a valid coupling straddles live GEO_AMR_INTERFACE cells only
# (asserted statically by checkCouplingMapPattern at SimInit), so a covered
# window is an invalid registration, not a runtime case. The formerly
# default-on carve and its C2F_CARVE/C2F_NO_CARVE opt knobs have no code
# left to gate; configuring with either set warns loudly here.
if(C2F_CARVE OR C2F_NO_CARVE)
    message(WARNING "C2F_CARVE/C2F_NO_CARVE were retired on 2026-08-23: the carve pre-pass is hard-removed (the ch7 band map-pattern assertion makes covered windows an invalid registration, not a runtime case). Remove the define from your configure line.")
endif()

# 2026-08-20 (Schönherr six-step cycle, plan ruling (d)): the H9
# time-centered first-fill experiment was HARD-REMOVED together with the
# legacy per-substep BVP refill -- the cycle now fills both AB frames once
# per cycle end with identical content, so there is no code left for the
# knob to gate. Configuring with C2F_H9 set warns loudly here.
if(C2F_H9)
    message(WARNING "C2F_H9 was retired on 2026-08-20 (Schönherr six-step cycle): the time-centered first fill and the per-substep BVP refill are hard-removed -- the cycle fills both frames once per cycle end with identical content. Remove -DC2F_H9 from your configure line.")
endif()

# optional C2F_EQ_ONLY debug switch: the fills write pure equilibrium
# (CM strain-rate moments and Lagrange non-equilibrium rescale are zeroed);
# experiment for the interface density-bias investigation
set(C2F_EQ_ONLY "" CACHE STRING "C2F equilibrium-only fill debug define for AMR targets (enable with ON)")

# optional C2F_DEV_ONLY trace-off debug switch: deviatoric-only strain-rate
# moments in the CM branch fill (trace of Pi subtracted; probe for the
# interface density-bias investigation)
set(C2F_DEV_ONLY "" CACHE STRING "C2F deviatoric-only strain fill debug define for AMR targets (enable with ON)")

# optional C2F_NORM_ONLY debug switch: keep only the diagonal deviatoric part
# of the non-equilibrium pressure tensor in the CM branch fill (zero the
# off-diagonal shear; probe for the interface density-bias investigation)
set(C2F_NORM_ONLY "" CACHE STRING "C2F normal-stress-only strain fill debug define for AMR targets (enable with ON)")

# optional C2F_SHEAR_ONLY debug switch: keep only the off-diagonal shear part
# of the non-equilibrium pressure tensor in the CM branch fill (zero the
# diagonals; closes the NORM/SHEAR partition of the bias attribution)
set(C2F_SHEAR_ONLY "" CACHE STRING "C2F shear-only strain fill debug define for AMR targets (enable with ON)")

# 2026-09-05 frame/cycle-timing probe round (suspects T2/T1 of the 2026-09-03
# probe synthesis for the parity-locked checkerboard artifact at the AMR
# coarse/fine interface; arm semantics and the measured verdicts are documented
# in the file docstring of include/lbm3d/d3q27/amr_coupling.h). All OFF by
# default: with every option unset the sim targets compile the verbatim HEAD
# text pinned by the bit-identity harness.
set(AMR_C2F_READ_DF_CUR "" CACHE STRING "T2a frame-timing probe: the C2F fill reads the coarse pre-step frame df_cur instead of the post-step df_out (enable with ON)")
set(AMR_F2C_READ_DF_CUR "" CACHE STRING "T2b frame-timing probe: the F2C skin feedback reads the fine pair-mid frame df_cur instead of the post-pair df_out (enable with ON)")
set(AMR_C2F_WRITE_DF_OUT "" CACHE STRING "T2c frame-timing probe: the C2F fill writes the frame the next fine substep does not consume (enable with ON)")
set(AMR_PASSIVE_BAND "" CACHE STRING "T1a cycle-timing probe: disarm the substep-1 widened extent (both substeps interior-only, frame-1 fill restored) (enable with ON)")
set(AMR_SUBSTEP2_C2F_REFILL "" CACHE STRING "T1b cycle-timing probe: refill the substep-1 frame's ghost rows between the fine substeps (enable with ON)")

# 2026-09-05 operator-parametrization probe (round 2b of the checkerboard
# investigation -- hypothesis and measured verdict in col_cum.h next to the
# pin, arm table in docs/AMR-seam-checkerboard-investigation.md §2): pin the
# three Part-I-parametrized relaxation rates omega3/omega4/omega5 to exactly
# 1 GLOBALLY in the D3Q27 cumulant collision (the Kutscher-2018 production
# rates family; the GLOBAL counterpart of the round-1 band-local
# AMR_BAND_OMEGA3 arm, with which stacking is a compile-time #error in
# col_cum.h). The arm exists only under USE_GEIER_CUM_2017 -- of the AMR
# targets only sim_AMR_ball compiles that mode, so it is a no-op define for
# sim_AMR/sim_AMR_channel (same precedent as AMR_BAND_OMEGA3). OFF by
# default: unset, the targets compile the verbatim HEAD text pinned by the
# bit-identity harness.
set(CUM_OMEGA345_ONE "" CACHE STRING "Round-2b operator-parametrization probe: pin omega3/omega4/omega5 to exactly 1 GLOBALLY in the D3Q27 cumulant collision (K2018 production rates family; exists only under USE_GEIER_CUM_2017 = sim_AMR_ball; mutually exclusive with AMR_BAND_OMEGA3) (enable with ON)")

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

    if(C2F_EQ_ONLY)
        target_compile_definitions(${target} PRIVATE C2F_EQ_ONLY)
    endif()

    if(C2F_DEV_ONLY)
        target_compile_definitions(${target} PRIVATE C2F_DEV_ONLY)
    endif()

    if(C2F_NORM_ONLY)
        target_compile_definitions(${target} PRIVATE C2F_NORM_ONLY)
    endif()

    if(C2F_SHEAR_ONLY)
        target_compile_definitions(${target} PRIVATE C2F_SHEAR_ONLY)
    endif()

    # 2026-09-05 frame/cycle-timing probe round (see the block above)
    if(AMR_C2F_READ_DF_CUR)
        target_compile_definitions(${target} PRIVATE AMR_C2F_READ_DF_CUR)
    endif()

    if(AMR_F2C_READ_DF_CUR)
        target_compile_definitions(${target} PRIVATE AMR_F2C_READ_DF_CUR)
    endif()

    if(AMR_C2F_WRITE_DF_OUT)
        target_compile_definitions(${target} PRIVATE AMR_C2F_WRITE_DF_OUT)
    endif()

    if(AMR_PASSIVE_BAND)
        target_compile_definitions(${target} PRIVATE AMR_PASSIVE_BAND)
    endif()

    if(AMR_SUBSTEP2_C2F_REFILL)
        target_compile_definitions(${target} PRIVATE AMR_SUBSTEP2_C2F_REFILL)
    endif()

    # 2026-09-05 operator-parametrization probe (see the block above)
    if(CUM_OMEGA345_ONE)
        target_compile_definitions(${target} PRIVATE CUM_OMEGA345_ONE)
    endif()

    # force language to HIP
    if(TNL_LBM_BUILD_HIP)
        set_source_files_properties(${target}.cu PROPERTIES LANGUAGE HIP)
    endif()
endforeach()
