#pragma once

#include "nlohmann/json.hpp"
#include <string>

namespace heat{
struct Input {
    int rows = 2000;
    int cols = 2000;
    int nsteps = 1000;
    int image_interval = 500;
    bool read_file = false;
    std::string fname = "";

    bool operator==(const Input &rhs) const {
        bool equal = true;
        equal &= rows == rhs.rows;
        equal &= cols == rhs.cols;
        equal &= nsteps == rhs.nsteps;
        equal &= image_interval == rhs.image_interval;
        equal &= read_file == rhs.read_file;
        equal &= fname == rhs.fname;

        return equal;
    }

    bool operator!=(const Input &rhs) const { return !(*this == rhs); }
};

void to_json(nlohmann::json &j, const Input &from);
void from_json(const nlohmann::json &j, Input &to);
Input read_input(const char *fname);
}
