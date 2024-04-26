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

    auto height = field.nx * parallel.size;
    auto width = field.ny;

    // array for MPI sends and receives
    auto tmp_mat = Matrix<double> (field.nx, field.ny); 

    if (0 == parallel.rank) {
        // Copy the inner data
        auto full_data = Matrix<double>(height, width);
        for (int i = 0; i < field.nx; i++)
            for (int j = 0; j < field.ny; j++) 
                 full_data(i, j) = field(i + 1, j + 1);
          
        // Receive data from other ranks
        for (int p = 1; p < parallel.size; p++) {
            MPI_Recv(tmp_mat.data(), field.nx * field.ny,
                     MPI_DOUBLE, p, 22, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            // Copy data to full array 
            for (int i = 0; i < field.nx; i++) 
                for (int j = 0; j < field.ny; j++) 
                     full_data(i + p * field.nx, j) = tmp_mat(i, j);
        }
        // Write out the data to a png file 
        std::ostringstream filename_stream;
        filename_stream << "heat_" << std::setw(4) << std::setfill('0') << iter << ".png";
        std::string filename = filename_stream.str();
        save_png(full_data.data(), height, width, filename.c_str(), 'c');
    } else {
        // Send data 
        for (int i = 0; i < field.nx; i++)
            for (int j = 0; j < field.ny; j++)
                tmp_mat(i, j) = field(i + 1, j + 1);

        MPI_Send(tmp_mat.data(), field.nx * field.ny,
                 MPI_DOUBLE, 0, 22, MPI_COMM_WORLD);
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
        int nx_full = 0;
        int ny_full = 0;
        std::stringstream(line) >> comment >> nx_full >> ny_full;

        // Read data to a vector
        std::vector<double> full_data;
        if (0 == parallel.rank) {
            std::istream_iterator<double> start(file);
            std::istream_iterator<double> end;
            full_data = std::vector<double>(start, end);
        }

        const auto [nx, ny] =
            Field::partition_domain(nx_full, ny_full, parallel.size);
        const auto num_values_per_rank = nx * ny;

        std::vector<double> my_data(num_values_per_rank);
        MPI_Scatter(full_data.data(), num_values_per_rank, MPI_DOUBLE,
                    my_data.data(), num_values_per_rank, MPI_DOUBLE, 0,
                    MPI_COMM_WORLD);

        // After this, possible to pass inner and nx to field constructor
        // So this could return nx, ny and my_data
        field.setup(nx_full, ny_full, parallel);

        // Copy to the array containing also boundaries
        for (int i = 0; i < field.nx; i++)
            for (int j = 0; j < field.ny; j++)
                field(i + 1, j + 1) = my_data[i * field.ny + j];

        // Set the boundary values
        for (int i = 0; i < field.nx + 2; i++) {
            // left boundary
            field(i, 0) = field(i, 1);
            // right boundary
            field(i, field.ny + 1) = field(i, field.ny);
        }

        for (int j = 0; j < field.ny + 2; j++) {
            // top boundary
            field.temperature(0, j) = field(1, j);
            // bottom boundary
            field.temperature(field.nx + 1, j) = field.temperature(field.nx, j);
        }

        return;
    } else {
        err_msg << "Could not open file \"" << filename << "\"";
    }

    throw std::runtime_error(err_msg.str());
}
