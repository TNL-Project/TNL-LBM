add_executable(test_outputdata integration/test_outputdata.cu)
target_link_libraries(test_outputdata PUBLIC TNL::TNL_LBM)

# force language to HIP
if(CMAKE_HIP_COMPILER)
    set_source_files_properties(integration/test_outputdata.cu PROPERTIES LANGUAGE HIP)
endif()
