add_subdirectory(GettingStarted)
add_subdirectory(LidDrivenCavity)
add_subdirectory(CollisionOperators)
add_subdirectory(StreamingSchemes)
add_subdirectory(MultiGPU)

configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/users-guide.md
    ${CMAKE_CURRENT_BINARY_DIR}/users-guide.md
    COPYONLY
)
