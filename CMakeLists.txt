cmake_minimum_required(VERSION 3.28)

# Set default CUDA architecture
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES "native")
endif()

project(TNL-LBM
    # FIXME: C is needed only for Adios2 https://github.com/ornladios/ADIOS2/issues/4082
    LANGUAGES CXX CUDA C
)

# Declare all CMake options for the project
option(TNL_LBM_EXPORT_INTERFACE_TARGETS "Instruct CMake to generate rules to export the interface target" ${PROJECT_IS_TOP_LEVEL})
option(TNL_LBM_ENABLE_MPI "Enable MPI support, if available" ON)

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

# Enforce (more or less) warning-free builds
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Werror -Wno-error=deprecated -Wno-error=deprecated-declarations -Wno-error=uninitialized")

if(CMAKE_CUDA_COMPILER_ID STREQUAL "NVIDIA")
    set(CMAKE_CUDA_FLAGS_RELWITHDEBINFO "${CMAKE_CUDA_FLAGS_RELWITHDEBINFO} --generate-line-info")
    if(NOT CMAKE_CXX_COMPILER_ID STREQUAL "NVHPC")
        # Enforce (more or less) warning-free builds for host code
        set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler -Werror -Xcompiler -Wno-error=deprecated -Xcompiler -Wno-error=deprecated-declarations")
    endif()
    # Force colorized output for nvcc's host compiler (assuming it is GCC...)
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler -fdiagnostics-color")
endif()

if(CMAKE_CUDA_COMPILER_ID STREQUAL "Clang")
    # Enforce (more or less) warning-free builds
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Werror -Wno-error=deprecated -Wno-error=deprecated-declarations -Wno-error=unknown-cuda-version")
    # workaround for Clang 15
    # https://github.com/llvm/llvm-project/issues/58491
    set(CMAKE_CUDA_FLAGS_DEBUG "-g -Xarch_device -g0")
endif()

# Disable GCC's infamous warnings (they produce many false positives)
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-maybe-uninitialized -Wno-stringop-overflow")
    if(CMAKE_CUDA_COMPILER_ID STREQUAL "NVIDIA")
        set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler -Wno-maybe-uninitialized -Wno-stringop-overflow")
    endif()
endif()

# Warn about redundant semicolons
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" OR
   CMAKE_CXX_COMPILER_ID STREQUAL "Clang" OR
   CMAKE_CXX_COMPILER_ID STREQUAL "AppleClang" OR
   CMAKE_CXX_COMPILER_ID STREQUAL "IntelLLVM")
    set( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wextra-semi" )
    if(CMAKE_CXX_COMPILER_ID STREQUAL "Clang" OR
       CMAKE_CXX_COMPILER_ID STREQUAL "AppleClang" OR
       CMAKE_CXX_COMPILER_ID STREQUAL "IntelLLVM")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wextra-semi-stmt")
    endif()
endif()

# Make cache variable so it can be used in downstream projects
set(TNL_LBM_INCLUDE_DIRS "${CMAKE_CURRENT_LIST_DIR}/include" CACHE INTERNAL "Directories where TNL-LBM headers are located")

# Create the exported targets
add_library(TNL_LBM INTERFACE)
# Create aliases to match exported targets
add_library(TNL::TNL_LBM ALIAS TNL_LBM)

if(TNL_LBM_EXPORT_INTERFACE_TARGETS)
    # Export the interface targets
    install(TARGETS TNL_LBM EXPORT "${PROJECT_NAME}Targets")
    # Install a CMake file for the interface target
    install(
        EXPORT "${PROJECT_NAME}Targets"
        NAMESPACE TNL::
        DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/${PROJECT_NAME}"
    )
    # Install the Config.cmake file
    include(CMakePackageConfigHelpers)
    configure_package_config_file(
        Config.cmake.in "${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}Config.cmake"
        INSTALL_DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/${PROJECT_NAME}"
    )
    install(FILES "${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}Config.cmake"
            DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/${PROJECT_NAME}"
    )
endif()

# Add the include directory to the interface
if(PROJECT_IS_TOP_LEVEL)
    set(TNL_LBM_SYSTEM "")
else()
    set(TNL_LBM_SYSTEM SYSTEM)
endif()
target_include_directories(
    TNL_LBM ${TNL_LBM_SYSTEM} INTERFACE
    $<BUILD_INTERFACE:${TNL_LBM_INCLUDE_DIRS}>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)

# Find or fetch dependencies
include(FetchContent)
FetchContent_Declare(fmt
    GIT_REPOSITORY https://github.com/fmtlib/fmt.git
    GIT_TAG 12.0.0
    SYSTEM
    FIND_PACKAGE_ARGS
)
set(FMT_INSTALL ON)  # needed for adding to the interface target
FetchContent_MakeAvailable(fmt)

FetchContent_Declare(spdlog
    GIT_REPOSITORY https://github.com/gabime/spdlog.git
    GIT_TAG v1.15.3
    SYSTEM
    FIND_PACKAGE_ARGS
)
set(SPDLOG_INSTALL ON)  # needed for adding to the interface target
set(SPDLOG_FMT_EXTERNAL ON)
FetchContent_MakeAvailable(spdlog)

FetchContent_Declare(nlohmann_json
    GIT_REPOSITORY https://github.com/nlohmann/json.git
    GIT_TAG v3.12.0
    SYSTEM
    FIND_PACKAGE_ARGS
)
set(JSON_Install ON)  # needed for adding to the interface target
set(JSON_SystemInclude ON)
FetchContent_MakeAvailable(nlohmann_json)

FetchContent_Declare(argparse
    GIT_REPOSITORY https://github.com/p-ranav/argparse.git
    GIT_TAG v3.2
    SYSTEM
    FIND_PACKAGE_ARGS
)
set(CMAKE_POLICY_DEFAULT_CMP0077 NEW)
set(ARGPARSE_INSTALL ON)  # needed for adding to the interface target
FetchContent_MakeAvailable(argparse)

FetchContent_Declare(magic_enum
    GIT_REPOSITORY https://github.com/Neargye/magic_enum
    GIT_TAG v0.9.7
    SYSTEM
    FIND_PACKAGE_ARGS
)
set(MAGIC_ENUM_OPT_INSTALL ON)  # needed for adding to the interface target
FetchContent_MakeAvailable(magic_enum)

FetchContent_Declare(TNL
    GIT_REPOSITORY https://gitlab.com/tnl-project/tnl.git
    # TNL does not have any releases
    GIT_TAG 9e7b0f44cde84d847e65dcab508785445b28cada
    EXCLUDE_FROM_ALL
)
set(TNL_EXPORT_INTERFACE_TARGETS ON)  # needed for adding to the interface target
FetchContent_MakeAvailable(TNL)

find_package(PNG REQUIRED)
find_package(OpenMP COMPONENTS CXX)
find_package(MPI COMPONENTS CXX)
find_package(ADIOS2 REQUIRED)

# Add dependencies to the interface target
target_link_libraries(TNL_LBM INTERFACE
    TNL::TNL
    PNG::PNG
    fmt::fmt
    spdlog::spdlog
    nlohmann_json::nlohmann_json
    argparse::argparse
    magic_enum::magic_enum
)
if(OPENMP_FOUND)
    target_compile_definitions(TNL_LBM INTERFACE "-DHAVE_OPENMP")
    target_link_libraries(TNL_LBM INTERFACE OpenMP::OpenMP_CXX)
    target_compile_options(TNL_LBM INTERFACE $<$<COMPILE_LANG_AND_ID:CUDA,Clang>: ${OpenMP_CXX_FLAGS} >)
    # nvcc needs -Xcompiler
    target_compile_options(TNL_LBM INTERFACE $<$<COMPILE_LANG_AND_ID:CUDA,NVIDIA>: -Xcompiler=-fopenmp >)
endif()
if(MPI_FOUND AND TNL_LBM_ENABLE_MPI)
    target_compile_definitions(TNL_LBM INTERFACE "-DHAVE_MPI")
    target_link_libraries(TNL_LBM INTERFACE MPI::MPI_CXX adios2::cxx11_mpi)
else()
    target_link_libraries(TNL_LBM INTERFACE MPI::MPI_CXX adios2::cxx11)
endif()
# Add special flags used in lbm3d
target_compile_definitions(TNL_LBM INTERFACE "-DUSE_CUDA")

# Add project subdirectories
add_subdirectory(include)
add_subdirectory(sim_NSE)
add_subdirectory(sim_NSE_ADE)
