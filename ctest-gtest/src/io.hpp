#pragma once
#include "nlohmann/json_fwd.hpp"
#include <string>
#include <tuple>
#include <vector>

struct Field;
struct ParallelData;

void write_field(const Field &field, const int iter,
                 const ParallelData &parallel);

void read_field(Field &field, const std::string &filename,
                const ParallelData &parallel);

namespace heat {
std::tuple<int, int, std::vector<double>>
read_field(const std::string &filename, int rank);

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
Input read_input(std::string &&fname, int rank);
}
