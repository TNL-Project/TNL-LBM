cmake_minimum_required(VERSION 3.28)

# Set default CUDA architecture
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES "native")
endif()

project(test-subproject LANGUAGES CXX CUDA)

# Require C++17, and disable compiler-specific extensions (if possible).
foreach(lang CXX CUDA)
    set(CMAKE_${lang}_STANDARD 17)
    set(CMAKE_${lang}_STANDARD_REQUIRED ON)
    set(CMAKE_${lang}_EXTENSIONS OFF)
endforeach()

# Set build flags for CXX
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Werror=vla")
set(CMAKE_CXX_FLAGS_DEBUG "-g")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")
set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "${CMAKE_CXX_FLAGS_RELEASE} ${CMAKE_CXX_FLAGS_DEBUG}")

# Set build flags for CUDA
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Wall")
set(CMAKE_CUDA_FLAGS_DEBUG "-g")
set(CMAKE_CUDA_FLAGS_RELEASE "-O3 -DNDEBUG")
set(CMAKE_CUDA_FLAGS_RELWITHDEBINFO "${CMAKE_CUDA_FLAGS_RELEASE} ${CMAKE_CUDA_FLAGS_DEBUG}")

# Warn about redundant semicolons
if(
    CMAKE_CXX_COMPILER_ID STREQUAL "GNU"
    OR CMAKE_CXX_COMPILER_ID STREQUAL "Clang"
    OR CMAKE_CXX_COMPILER_ID STREQUAL "AppleClang"
    OR CMAKE_CXX_COMPILER_ID STREQUAL "IntelLLVM"
)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wextra-semi")
    if(
        CMAKE_CXX_COMPILER_ID STREQUAL "Clang"
        OR CMAKE_CXX_COMPILER_ID STREQUAL "AppleClang"
        OR CMAKE_CXX_COMPILER_ID STREQUAL "IntelLLVM"
    )
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wextra-semi-stmt")
    endif()
endif()

# Add dependencies
include(FetchContent)
FetchContent_Declare(TNL-LBM GIT_REPOSITORY https://gitlab.com/tnl-project/tnl-lbm.git GIT_TAG main EXCLUDE_FROM_ALL)
FetchContent_MakeAvailable(TNL-LBM)

# Add test target
add_executable(main main.cu)
target_link_libraries(main PUBLIC TNL::TNL_LBM)
