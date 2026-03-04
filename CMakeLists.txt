cmake_minimum_required(VERSION 3.21)
project(msplat LANGUAGES CXX OBJCXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})

# Version
file(READ "VERSION" APP_VERSION)
string(STRIP "${APP_VERSION}" APP_VERSION)
set(GIT_REV "")
execute_process(COMMAND git rev-parse --short HEAD
    WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
    OUTPUT_VARIABLE GIT_REV ERROR_QUIET)
string(REGEX REPLACE "\n$" "" GIT_REV "${GIT_REV}")
if(NOT "${GIT_REV}" STREQUAL "")
    set(DAPP_VERSION "${APP_VERSION} (git ${GIT_REV})")
    set(DAPP_REVISION "${GIT_REV}")
else()
    set(DAPP_VERSION "${APP_VERSION}")
    set(DAPP_REVISION "dev")
endif()
message(STATUS "msplat ${DAPP_VERSION}")

add_compile_options("-DAPP_VERSION=\"${DAPP_VERSION}\"")
add_compile_options("-DAPP_REVISION=\"${DAPP_REVISION}\"")

# --- Dependencies ---

set(CMAKE_POLICY_DEFAULT_CMP0077 NEW)
if(CMAKE_VERSION VERSION_GREATER_EQUAL "3.24.0")
    cmake_policy(SET CMP0135 NEW)
endif()

include(FetchContent)
FetchContent_Declare(nlohmann_json
    URL https://github.com/nlohmann/json/archive/refs/tags/v3.11.3.zip)
set(NANOFLANN_BUILD_EXAMPLES OFF)
set(NANOFLANN_BUILD_TESTS OFF)
FetchContent_Declare(nanoflann
    URL https://github.com/jlblancoc/nanoflann/archive/refs/tags/v1.5.5.zip)
FetchContent_Declare(cli11
    URL https://github.com/CLIUtils/CLI11/archive/refs/tags/v2.4.2.zip)
FetchContent_MakeAvailable(nlohmann_json nanoflann cli11)

# Metal frameworks
find_library(FOUNDATION_LIBRARY Foundation REQUIRED)
find_library(METAL_FRAMEWORK Metal REQUIRED)
find_library(METALKIT_FRAMEWORK MetalKit REQUIRED)
find_library(MPS_FRAMEWORK MetalPerformanceShaders REQUIRED)

find_library(IMAGEIO_FRAMEWORK ImageIO REQUIRED)
find_library(COREGRAPHICS_FRAMEWORK CoreGraphics REQUIRED)

# --- Compile Metal shaders ---

set(METAL_SOURCE ${CMAKE_SOURCE_DIR}/core/metal/msplat_metal.metal)
configure_file(${METAL_SOURCE} ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/msplat_metal.metal COPYONLY)
add_custom_command(
    OUTPUT ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/default.metallib
    COMMAND xcrun -sdk macosx metal -O3 -c
        ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/msplat_metal.metal
        -o ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/msplat_metal.air
    COMMAND xcrun -sdk macosx metallib
        ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/msplat_metal.air
        -o ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/default.metallib
    COMMAND rm -f ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/msplat_metal.air
    COMMAND rm -f ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/msplat_metal.metal
    DEPENDS ${METAL_SOURCE}
    COMMENT "Compiling Metal kernels → default.metallib"
)
add_custom_target(metallib ALL
    DEPENDS ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/default.metallib)

# --- Core static library ---

add_library(msplat_core STATIC
    core/metal/msplat_metal.mm
    core/src/msplat_api.mm
    core/src/model.cpp
    core/src/input_data.cpp
    core/src/loaders/load_colmap.cpp
    core/src/loaders/load_nerfstudio.cpp
    core/src/loaders/load_polycam.cpp
    core/src/loaders/load_ply.cpp
    core/src/loaders/image_io.cpp
    core/src/loaders/save_gaussians.cpp
    core/src/ssim.cpp
)
target_include_directories(msplat_core PUBLIC
    ${CMAKE_SOURCE_DIR}/core/include
    ${CMAKE_SOURCE_DIR}/core/metal
)
target_link_libraries(msplat_core PUBLIC
    ${FOUNDATION_LIBRARY}
    ${METAL_FRAMEWORK}
    ${METALKIT_FRAMEWORK}
    ${MPS_FRAMEWORK}
    ${IMAGEIO_FRAMEWORK}
    ${COREGRAPHICS_FRAMEWORK}
    nlohmann_json::nlohmann_json
    nanoflann::nanoflann
)
target_compile_definitions(msplat_core PRIVATE USE_MPS)
add_dependencies(msplat_core metallib)

# --- CLI binary ---

add_executable(msplat cli/msplat.cpp)
target_link_libraries(msplat PRIVATE
    msplat_core
    CLI11::CLI11
    pthread
)
target_compile_definitions(msplat PRIVATE USE_MPS)
install(TARGETS msplat DESTINATION bin)

# --- Python bindings (optional, enabled when building via scikit-build-core) ---

option(MSPLAT_BUILD_PYTHON "Build nanobind Python bindings" OFF)

if(MSPLAT_BUILD_PYTHON)
    find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)
    find_package(nanobind CONFIG REQUIRED)

    nanobind_add_module(_core python/bindings.cpp)
    target_link_libraries(_core PRIVATE
        msplat_core
        pthread
    )
    target_compile_definitions(_core PRIVATE USE_MPS)

    install(TARGETS _core LIBRARY DESTINATION msplat)
    install(FILES ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/default.metallib DESTINATION msplat)
endif()
