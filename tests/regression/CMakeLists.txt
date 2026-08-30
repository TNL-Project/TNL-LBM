# Dedicated 3-level nesting mock simulation for the ParaView end-to-end
# nesting arm (tests/integration/test_amr_paraview.py, the amr-nlevel-nesting
# plan's commit D): sim_AMR/sim_AMR_channel hardcode their region specs, so
# this compact driver produces the nested-region VTKHDF frames the e2e arm
# asserts. Single-pattern build (the e2e arm drives the default AB build).
add_executable(test_amr_nesting_sim test_amr_nesting_sim.cu)
target_link_libraries(test_amr_nesting_sim PUBLIC TNL::TNL_LBM)

# force language to HIP
if(TNL_LBM_BUILD_HIP)
    set_source_files_properties(test_amr_nesting_sim.cu PROPERTIES LANGUAGE HIP)
endif()
