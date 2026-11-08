//
// Created by edinj on 2026-11-08.
//
export module tensor;
import std;

export enum class DType : std::uint8_t {
    F32,
    F16,
    BF16,
    F64,
    I64,
    I32,
    I16,
    I8,
    U8,
    Bool,
};

// Bytes per element
export constexpr std::size_t size_of(const DType t) {
    switch (t) {
    case DType::F64:
    case DType::I64:
        return 8;
    case DType::F32:
    case DType::I32:
        return 4;
    case DType::F16:
    case DType::BF16:
    case DType::I16:
        return 2;
    case DType::I8:
    case DType::U8:
    case DType::Bool:
        return 1;
    }
    std::unreachable();
}



export struct TensorView {
    DType dtype;
    std::vector<std::size_t> shape;
    std::span<const std::byte> data; // points into tensor_container_
};

struct Block {
    TensorView ln_1_w, ln_1_b;
    TensorView attn_w, attn_b, attn_proj_w, attn_proj_b;
    TensorView ln_2_w, ln_2_b;
    TensorView mlp_fc_w, mlp_fc_b, mlp_proj_w, mlp_proj_b;
};

struct GPT2Weights {
    TensorView wte, wpe;
    std::vector<Block> h; // 12 for gpt2-small
    TensorView ln_f_w, ln_f_b;
};