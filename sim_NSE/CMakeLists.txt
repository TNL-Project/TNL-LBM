set(TARGETS
        sim_1
        sim_2
        sim_3
        sim_IBM1
        sim_IBM2
        sim_IBM3
        sim_IBM4
        sim_bump_NASA
        sim_schafer_turek
        sim_poiseuille
)

foreach(target IN ITEMS ${TARGETS})
    add_executable(${target} ${target}.cu)
    target_link_libraries(${target} PUBLIC TNL::TNL_LBM)
endforeach()
