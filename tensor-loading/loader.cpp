//
// Created by edinj on 2026-11-08.
//
import json;
import std;
import errors;
import tensor;

namespace {

// "F32" -> DType::F32, using the safetensors spelling
DType parse_dtype(const std::string_view s) {
    if (s == "F32")
        return DType::F32;
    if (s == "F16")
        return DType::F16;
    if (s == "BF16")
        return DType::BF16;
    if (s == "F64")
        return DType::F64;
    if (s == "I64")
        return DType::I64;
    if (s == "I32")
        return DType::I32;
    if (s == "I16")
        return DType::I16;
    if (s == "I8")
        return DType::I8;
    if (s == "U8")
        return DType::U8;
    if (s == "BOOL")
        return DType::Bool;
    throw std::invalid_argument{"Unknown dtype: " + std::string(s)};
}

struct TensorInfo {
    std::string dtype; // "F32", "F16", ...
    std::vector<std::size_t> shape;
    std::size_t begin, end; // data_offsets, relative to container_
};

class SafeTensorsLoader {
    std::vector<std::byte> tensor_bytes_container_;
    std::string filepath_;
    std::uint64_t header_len_ = 0;
    std::streamsize total_len_ = 0;
    std::streamsize tensors_data_len_ = 0;
    std::unordered_map<std::string, TensorInfo> tensors_;
    int json_indent_level_ = 2;
    bool print_header_ = false;
    nlohmann::json header_ = __null;

public:
    explicit SafeTensorsLoader(std::string filepath,
                               const int json_indent_level = 2,
                               const bool print_header = false)
        : filepath_{std::move(filepath)}, json_indent_level_{json_indent_level},
          print_header_{print_header} {
        this->loadTensors();
    }
    explicit SafeTensorsLoader() = default;
    void loadTensors() {
        std::ifstream f(filepath_, std::ios::binary);
        try {
            if (!f.good()) {
                throw std::runtime_error{"Cannot open file: " + filepath_};
            }

            // Total tensor file byte length
            f.seekg(0, std::ios::end);
            total_len_ = f.tellg();
            f.seekg(0, std::ios::beg);
            std::println("Total length is {} bytes", total_len_);

            // Header length
            f.read(reinterpret_cast<char *>(&header_len_), sizeof header_len_);
            std::println("Header is {} bytes", header_len_);

            // Header pretty JSON printout
            std::string raw_json(header_len_, '\0');
            f.read(raw_json.data(), static_cast<std::streamsize>(header_len_));
            try {
                header_ = nlohmann::json::parse(raw_json);
                if (print_header_)
                    std::println("{}", header_.dump(json_indent_level_));

                // Parse the header
                for (const auto &item : header_.items()) {
                    const auto &name = item.key();
                    const auto &v = item.value();
                    if (name == "__metadata__")
                        continue; // optional string->string map, not a tensor
                    auto off = v["data_offsets"];
                    tensors_.emplace(
                        name,
                        TensorInfo{
                            .dtype = v["dtype"].get<std::string>(),
                            .shape = v["shape"].get<std::vector<std::size_t>>(),
                            .begin = off[0].get<std::size_t>(),
                            .end = off[1].get<std::size_t>()});
                }

            } catch (const std::exception &e) {
                std::string message = "Could not capture SafeTensors header: " +
                                      std::string(e.what());
                std::println("{}", message);
                throw;
            }

            // Tensors-only length
            tensors_data_len_ = total_len_ - header_len_ - sizeof(header_len_);

            // Load container
            tensor_bytes_container_.resize(tensors_data_len_);
            auto tensors_start_ = header_len_ + sizeof(header_len_);
            f.seekg(static_cast<long>(tensors_start_), std::ios::beg);
            f.read(reinterpret_cast<char *>(tensor_bytes_container_.data()),
                   tensors_data_len_);
            if (f.gcount() != tensors_data_len_)
                throw std::runtime_error{"Truncated tensor data"};
        } catch (std::exception &e) {
            std::throw_with_nested(
                std::runtime_error{"Cannot read file: " + filepath_});
        }
    }
    TensorView getTensor(const std::string &name) {
        // Seek through the unordered map
        auto tensor_it = tensors_.find(name);
        if (tensor_it == tensors_.end()) {
            throw std::runtime_error{"Could not find tensor \"" + name + "\""};
        }
        const TensorInfo &tensor_info = tensor_it->second; // Getting the value
        // Sanity check for tensor data lengths
        if (tensor_info.end < tensor_info.begin ||
            tensor_info.end > tensor_bytes_container_.size())
            throw std::runtime_error{"Bad data_offsets for " + name};

        // Get a pointer to the relevant data start and length via slicing
        auto tensor_data =
            std::span<const std::byte>{tensor_bytes_container_}.subspan(
                tensor_info.begin, tensor_info.end - tensor_info.begin);

        return TensorView{parse_dtype(tensor_info.dtype), tensor_info.shape,
                          tensor_data};
    }
};
} // namespace

int main(int argc, char **argv) {
    // Path from the command line if given, otherwise the one CMake baked in.
    const std::string filepath = argc > 1 ? argv[1] : CRISPY_DEFAULT_MODEL;

    try {
        SafeTensorsLoader loader{filepath, 2, true};

        // 1. A known tensor has the right dtype, shape and byte size
        auto [dtype, shape, data] = loader.getTensor("wte.weight");
        if (dtype != DType::F32)
            throw std::runtime_error{"wte: wrong dtype"};
        if (shape != std::vector<std::size_t>{50257, 768})
            throw std::runtime_error{"wte: wrong shape"};
        if (data.size() != 50257 * 768 * size_of(DType::F32))
            throw std::runtime_error{"wte: wrong byte size"};

        // 2. The bytes look like real floats: print the first few
        std::array<float, 5> first{};
        std::memcpy(first.data(), data.data(), sizeof first);
        std::println("wte[0][:5] = {}", first);

        // 3. A missing name throws
        try {
            loader.getTensor("does.not.exist");
            throw std::logic_error{"missing tensor did not throw"};
        } catch (const std::runtime_error &) {
            // expected
        }

        std::println("smoke test OK");
    } catch (const std::exception &e) {
        print_exception(e);
    }
}