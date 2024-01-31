cmake_minimum_required(VERSION 3.28)

# Set default CUDA architecture
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES "native")
endif()

project(TNL-LBM LANGUAGES CXX)

# Require C++17, and disable compiler-specific extensions (if possible).
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Set build flags for CXX
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread -Wall -Werror=vla")
set(CMAKE_CXX_FLAGS_DEBUG "-g")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")
set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "${CMAKE_CXX_FLAGS_RELEASE} ${CMAKE_CXX_FLAGS_DEBUG}")

# TODO
# Enforce (more or less) warning-free builds
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Werror -Wno-error=deprecated -Wno-error=deprecated-declarations -Wno-error=uninitialized")

# Warn about redundant semicolons
if( CMAKE_CXX_COMPILER_ID STREQUAL "GNU" OR
    CMAKE_CXX_COMPILER_ID STREQUAL "Clang" OR
    CMAKE_CXX_COMPILER_ID STREQUAL "AppleClang"
)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wextra-semi")
    if( CMAKE_CXX_COMPILER_ID STREQUAL "Clang" OR
        CMAKE_CXX_COMPILER_ID STREQUAL "AppleClang"
    )
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wextra-semi-stmt")
    endif()
endif()

include(CheckLanguage)

check_language(CUDA)
if(CMAKE_CUDA_COMPILER)
    enable_language(CUDA)

    # Require C++17, and disable compiler-specific extensions (if possible).
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    set(CMAKE_CUDA_EXTENSIONS OFF)

    # Set build flags for CUDA
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Wall")
    set(CMAKE_CUDA_FLAGS_DEBUG "-g")
    set(CMAKE_CUDA_FLAGS_RELEASE "-O3 -DNDEBUG")
    set(CMAKE_CUDA_FLAGS_RELWITHDEBINFO "${CMAKE_CUDA_FLAGS_RELEASE} ${CMAKE_CUDA_FLAGS_DEBUG}")
    if(CMAKE_CUDA_COMPILER_ID STREQUAL "NVIDIA")
        set(CMAKE_CUDA_FLAGS_RELWITHDEBINFO "${CMAKE_CUDA_FLAGS_RELWITHDEBINFO} --generate-line-info")
    endif()
endif()

check_language(HIP)
if(CMAKE_HIP_COMPILER)
    enable_language(HIP)

    # Require C++17, and disable compiler-specific extensions (if possible).
    set(CMAKE_HIP_STANDARD 17)
    set(CMAKE_HIP_STANDARD_REQUIRED ON)
    set(CMAKE_HIP_EXTENSIONS OFF)

    # Set build flags for HIP
    set( CMAKE_HIP_FLAGS "${CMAKE_HIP_FLAGS} -Wall" )
    # NOTE: cmake mirrors the hipcc.pl script and passes some flags to rocm-llvm
    #       (--offload-arch=gfx803 -mllvm -amdgpu-early-inline-all=true -mllvm -amdgpu-function-calls=false)
    #       which do not work with -O0 (leads to Memory access fault), hence -O1
    set( CMAKE_HIP_FLAGS_DEBUG "-O1 -g -DHIP_ENABLE_PRINTF" )
    set( CMAKE_HIP_FLAGS_RELEASE "-O3 -DNDEBUG" )
    set( CMAKE_HIP_FLAGS_RELWITHDEBINFO "${CMAKE_HIP_FLAGS_RELEASE} ${CMAKE_HIP_FLAGS_DEBUG}" )

    # TODO
    # enforce (more or less) warning-free builds
    #set( CMAKE_HIP_FLAGS "${CMAKE_HIP_FLAGS} -Werror -Wno-error=deprecated -Wno-error=deprecated-declarations" )
endif()

# Set the project's include path
include_directories(include)

# Add dependencies
include(FetchContent)
FetchContent_Declare(fmt
    GIT_REPOSITORY https://github.com/fmtlib/fmt.git
    GIT_TAG master
    FIND_PACKAGE_ARGS
)
FetchContent_MakeAvailable(fmt)

FetchContent_Declare(spdlog
    GIT_REPOSITORY https://github.com/gabime/spdlog.git
    GIT_TAG master
    FIND_PACKAGE_ARGS
)
FetchContent_MakeAvailable(spdlog)

FetchContent_Declare(TNL
    GIT_REPOSITORY https://gitlab.com/tnl-project/tnl.git
    GIT_TAG main
    EXCLUDE_FROM_ALL
)
FetchContent_MakeAvailable(TNL)

find_package(PNG REQUIRED)
find_package(OpenMP COMPONENTS CXX)
find_package(MPI COMPONENTS CXX)

# Add project subdirectories
add_subdirectory(include)
add_subdirectory(sim_NSE)
add_subdirectory(sim_NSE_ADE)
