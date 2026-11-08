add_executable(safetensors_loader loader.cpp)
target_compile_options(safetensors_loader PRIVATE -Wall -Wextra
    $<$<CONFIG:Debug>:-fsanitize=address,undefined>)
target_link_options(safetensors_loader PRIVATE
    $<$<CONFIG:Debug>:-fsanitize=address,undefined>)
# Absolute default path, so the binary works no matter which directory it runs from.
target_compile_definitions(safetensors_loader PRIVATE
    CRISPY_DEFAULT_MODEL="${PROJECT_SOURCE_DIR}/models/gpt2/model.safetensors")

# Json pretty printing
include(FetchContent)
FetchContent_Declare(json
        URL https://github.com/nlohmann/json/releases/download/v3.12.0/json.tar.xz)
FetchContent_MakeAvailable(json)

target_sources(safetensors_loader PRIVATE FILE_SET CXX_MODULES FILES json.cpp)
target_link_libraries(safetensors_loader PRIVATE
    nlohmann_json::nlohmann_json
    crispy_common
    crispy_tensor
)
