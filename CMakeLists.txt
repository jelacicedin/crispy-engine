cmake_minimum_required(VERSION 4.4)

# `import std` is still experimental in CMake; this key is specific to CMake 4.4
# and must be updated if CMake reports it as unknown after an upgrade.
set(CMAKE_EXPERIMENTAL_CXX_IMPORT_STD "f35a9ac6-8463-4d38-8eec-5d6008153e7d")
project(crispy_engine LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_MODULE_STD ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

add_subdirectory(common)
add_subdirectory(tensor)
add_subdirectory(tensor-loading)
