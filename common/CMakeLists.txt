# Shared helpers used across the engine's targets.
add_library(crispy_common)
target_sources(crispy_common PUBLIC FILE_SET CXX_MODULES FILES errors.cpp)
target_compile_options(crispy_common PRIVATE -Wall -Wextra)
