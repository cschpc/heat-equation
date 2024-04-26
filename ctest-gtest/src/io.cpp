/* I/O related functions for heat equation solver */

#include <fstream>
#include <iomanip>
#include <iterator>
#include <mpi.h>
#include <stdexcept>
#include <string>

#include "matrix.hpp"
#include "heat.hpp"
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
        for (int i = 0; i < field.num_rows; i++)
            for (int j = 0; j < field.num_cols; j++)
                full_data(i, j) = field(i + 1, j + 1);
          
        // Receive data from other ranks
        for (int p = 1; p < parallel.size; p++) {
            MPI_Recv(tmp_mat.data(), field.num_rows * field.num_cols,
                     MPI_DOUBLE, p, 22, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            // Copy data to full array
            for (int i = 0; i < field.num_rows; i++)
                for (int j = 0; j < field.num_cols; j++)
                    full_data(i + p * field.num_rows, j) = tmp_mat(i, j);
        }
        // Write out the data to a png file 
        std::ostringstream filename_stream;
        filename_stream << "heat_" << std::setw(4) << std::setfill('0') << iter << ".png";
        std::string filename = filename_stream.str();
        save_png(full_data.data(), height, width, filename.c_str(), 'c');
    } else {
        // Send data
        for (int i = 0; i < field.num_rows; i++)
            for (int j = 0; j < field.num_cols; j++)
                tmp_mat(i, j) = field(i + 1, j + 1);

        MPI_Send(tmp_mat.data(), field.num_rows * field.num_cols, MPI_DOUBLE, 0,
                 22, MPI_COMM_WORLD);
    }
}

// Read the initial temperature distribution from a file
void read_field(Field &field, const std::string &filename,
                const ParallelData &parallel) {
    std::stringstream err_msg;

    std::ifstream file(filename);
    if (file.is_open()) {
        // Read the header
        std::string line;
        std::getline(file, line);

        std::string comment;
        int num_rows_global = 0;
        int num_cols_global = 0;
        std::stringstream(line) >> comment >> num_rows_global >>
            num_cols_global;

        // Read data to a vector
        std::vector<double> full_data;
        if (0 == parallel.rank) {
            std::istream_iterator<double> start(file);
            std::istream_iterator<double> end;
            full_data = std::vector<double>(start, end);
        }

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
        for (int i = 0; i < field.num_rows; i++)
            for (int j = 0; j < field.num_cols; j++)
                field(i + 1, j + 1) = my_data[i * field.num_cols + j];

        // Set the boundary values
        for (int i = 0; i < field.num_rows + 2; i++) {
            // left boundary
            field(i, 0) = field(i, 1);
            // right boundary
            field(i, field.num_cols + 1) = field(i, field.num_cols);
        }

        for (int j = 0; j < field.num_cols + 2; j++) {
            // top boundary
            field.temperature(0, j) = field(1, j);
            // bottom boundary
            field.temperature(field.num_rows + 1, j) =
                field.temperature(field.num_rows, j);
        }

        return;
    } else {
        err_msg << "Could not open file \"" << filename << "\"";
    }

    throw std::runtime_error(err_msg.str());
}
