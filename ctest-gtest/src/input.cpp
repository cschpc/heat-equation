#include <filesystem>
#include <fstream>
#include <iostream>

#include "input.hpp"

namespace heat {
void to_json(nlohmann::json &j, const Input &from) {
    j = nlohmann::json{
        {"rows", from.rows},           {"cols", from.cols},
        {"nsteps", from.nsteps},       {"image_interval", from.image_interval},
        {"read_file", from.read_file}, {"fname", from.fname},
    };
}

void from_json(const nlohmann::json &j, Input &to) {
    j.at("rows").get_to(to.rows);
    j.at("cols").get_to(to.cols);
    j.at("nsteps").get_to(to.nsteps);
    j.at("image_interval").get_to(to.image_interval);
    j.at("read_file").get_to(to.read_file);
    j.at("fname").get_to(to.fname);
}

// TODO: move to io.cpp
// Take ParallelData and write only with rank 0
Input read_input(const char *fname) {
    Input input = {};

    if (fname != nullptr) {
        const auto path = std::filesystem::path(fname);
        if (std::filesystem::exists(path)) {
            std::fstream file(path, std::ios::in);
            if (file.is_open()) {
                nlohmann::json j;
                try {
                    file >> j;
                    input = j.get<Input>();
                    std::cout << "Using input configuration from " << path
                              << std::endl;
                } catch (const nlohmann::json::exception &e) {
                    std::cerr << "Error parsing input with filename '" << fname
                              << "'. Exception thrown: " << e.what()
                              << std::endl;
                }
            } else {
                std::cerr << "Could not open file at path: " << path
                          << std::endl;
            }
        } else {
            std::cerr << "Cannot read input from non-existent path: " << path
                      << std::endl;
        }
    } else {
        std::cout << "No input file given, using default input" << std::endl;
    }

    return input;
}
}
