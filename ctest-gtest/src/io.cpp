/* I/O related functions for heat equation solver */

#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <mpi.h>
#include <sstream>
#include <stdexcept>
#include <tuple>

#include "field.hpp"
#include "io.hpp"
#include "matrix.hpp"
#include "parallel.hpp"
#include "pngwriter.h"

// Write a picture of the temperature field
void write_field(const Field &field, const int iter,
                 const ParallelData &parallel) {

    auto height = field.num_rows * parallel.size;
    auto width = field.num_cols;

    // array for MPI sends and receives
    auto tmp_mat = Matrix<double>(field.num_rows, field.num_cols);

    if (0 == parallel.rank) {
        // Copy the inner data
        auto full_data = Matrix<double>(height, width);
        for (int i = 0; i < field.num_rows; i++) {
            for (int j = 0; j < field.num_cols; j++) {
                full_data(i, j) = field(i, j);
            }
        }

        // Receive data from other ranks
        for (int p = 1; p < parallel.size; p++) {
            MPI_Recv(tmp_mat.data(), field.num_rows * field.num_cols,
                     MPI_DOUBLE, p, 22, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            // Copy data to full array
            for (int i = 0; i < field.num_rows; i++) {
                for (int j = 0; j < field.num_cols; j++) {
                    full_data(i + p * field.num_rows, j) = tmp_mat(i, j);
                }
            }
        }
        // Write out the data to a png file 
        std::ostringstream filename_stream;
        filename_stream << "heat_" << std::setw(4) << std::setfill('0') << iter << ".png";
        std::string filename = filename_stream.str();
        save_png(full_data.data(), height, width, filename.c_str(), 'c');
    } else {
        // Send data
        for (int i = 0; i < field.num_rows; i++) {
            for (int j = 0; j < field.num_cols; j++) {
                tmp_mat(i, j) = field(i, j);
            }
        }

        MPI_Send(tmp_mat.data(), field.num_rows * field.num_cols, MPI_DOUBLE, 0,
                 22, MPI_COMM_WORLD);
    }
}

// Read the initial temperature distribution from a file
void read_field(Field &field, const std::string &filename,
                const ParallelData &parallel) {
    auto [num_rows_global, num_cols_global, full_data] =
        heat::read_field(filename, parallel.rank);
    const auto [num_rows, num_cols] = Field::partition_domain(
        num_rows_global, num_cols_global, parallel.size);
    const auto num_values_per_rank = num_rows * num_cols;

    std::vector<double> my_data(num_values_per_rank);
    MPI_Scatter(full_data.data(), num_values_per_rank, MPI_DOUBLE,
                my_data.data(), num_values_per_rank, MPI_DOUBLE, 0,
                MPI_COMM_WORLD);

    // After this, possible to pass inner and num_rows to field constructor
    // So this could return num_rows, num_cols and my_data
    field.setup(num_rows_global, num_cols_global, parallel);

    // Copy to the array containing also boundaries
    for (int i = 0; i < field.num_rows; i++) {
        for (int j = 0; j < field.num_cols; j++) {
            field(i, j) = my_data[i * field.num_cols + j];
        }
    }

    // Set the boundary values
    for (int i = -1; i < field.num_rows + 1; i++) {
        // left boundary
        field(i, -1) = field(i, 0);
        // right boundary
        field(i, field.num_cols) = field(i, field.num_cols - 1);
    }

    // top boundary
    if (0 == parallel.rank) {
        for (int j = -1; j < field.num_cols + 1; j++) {
            field(-1, j) = field(0, j);
        }
    }

    // bottom boundary
    if (parallel.rank == parallel.size - 1) {
        for (int j = -1; j < field.num_cols + 1; j++) {
            field(field.num_rows, j) = field(field.num_rows - 1, j);
        }
    }
}

namespace heat {
std::tuple<int, int, std::vector<double>>
read_field(const std::string &filename, int rank) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::stringstream err_msg;
        err_msg << "Could not open file \"" << filename << "\"";
        throw std::runtime_error(err_msg.str());
    }

    // Read the header
    std::string line;
    std::getline(file, line);

    std::string comment;
    int num_rows = 0;
    int num_cols = 0;
    std::stringstream(line) >> comment >> num_rows >> num_cols;

    // Read data to a vector
    std::vector<double> full_data;
    if (0 == rank) {
        std::istream_iterator<double> start(file);
        std::istream_iterator<double> end;
        full_data = std::vector<double>(start, end);
    }

    return std::make_tuple(num_rows, num_cols, full_data);
}

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

Input read_input(const char *fname, int rank) {
    std::stringstream ess;
    if (fname == nullptr) {
        ess << "Filename is a nullptr";
        throw std::runtime_error(ess.str());
    }

    if (strlen(fname) == 0) {
        if (rank == 0) {
            std::cout << "Using default input" << std::endl;
        }
        return Input{};
    }

    const auto path = std::filesystem::path(fname);
    if (not std::filesystem::exists(path)) {
        ess << "Non-existent path: " << path;
        throw std::runtime_error(ess.str());
    }

    std::fstream file(path, std::ios::in);
    if (not file.is_open()) {
        ess << "Could not open file at " << path;
        throw std::runtime_error(ess.str());
    }

    if (rank == 0) {
        std::cout << "Reading input from " << path << std::endl;
    }
    nlohmann::json j;
    file >> j;
    return j.get<Input>();
}
} // namespace heat
