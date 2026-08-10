add_executable(test_outputdata test_outputdata.cu)
target_link_libraries(test_outputdata PUBLIC TNL::TNL_LBM)

# force language to HIP
if(TNL_LBM_BUILD_HIP)
    set_source_files_properties(test_outputdata.cu PROPERTIES LANGUAGE HIP)
endif()
