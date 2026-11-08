//
// Created by edinj on 2026-11-08.
//
export module errors;
import std;

export void print_exception(const std::exception &e, int level = 0) {
    std::println("{}{}", std::string(level * 2, ' '), e.what());
    try {
        std::rethrow_if_nested(e);
    } catch (const std::exception &inner) {
        print_exception(inner, level + 1);
    }
}