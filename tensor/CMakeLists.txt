add_library(crispy_tensor)
target_sources(crispy_tensor PUBLIC FILE_SET CXX_MODULES FILES tensor.cpp)
target_compile_options(crispy_tensor PRIVATE -Wall -Wextra)
