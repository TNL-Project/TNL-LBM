set(TARGETS
        sim_adjoint
        sim_pseudomeasure
)

foreach(target IN ITEMS ${TARGETS})
    add_executable(${target} ${target}.cu)
    target_link_libraries(${target} PUBLIC TNL::TNL_LBM)
endforeach()
