/* I/O related functions for heat equation solver */

#include <string>
#include <iomanip> 
#include <fstream>
#include <string>
#include <mpi.h>

#include "matrix.hpp"
#include "heat.hpp"
#include "pngwriter.h"

// Write a picture of the temperature field
void write_field(Field& field, const int iter, const ParallelData& parallel)
{

    auto height = field.nx_full;
    auto width = field.ny_full;
    auto length = field.nz_full;

    if (0 == parallel.rank) {
        // Copy the inner data
        auto full_data = Matrix<double>(height, width, length);
        for (int i = 0; i < field.nx; i++)
            for (int j = 0; j < field.ny; j++) 
              for (int k = 0; k < field.nz; k++) 
                 full_data(i, j, k) = field(i + 1, j + 1, k + 1);
          
        // Receive data from other ranks
        int coords[3];
        for (int p = 1; p < parallel.size; p++) {
            MPI_Cart_coords(parallel.comm, p, 3, coords);
            int ix = coords[0] * field.nx;
            int iy = coords[1] * field.ny;
            int iz = coords[2] * field.nz;
            MPI_Recv(full_data.data(ix, iy, iz), 1, parallel.subarraytype, p, 22,
                     parallel.comm, MPI_STATUS_IGNORE);
        }
        // Write out the middle slice of data to a png file 
        std::ostringstream filename_stream;
        filename_stream << "heat_" << std::setw(4) << std::setfill('0') << iter << ".png";
        std::string filename = filename_stream.str();
        save_png(full_data.data(0, 0, 0), height, width, filename.c_str(), 'c');
    } else {
        // Send data 
        MPI_Send(field.temperature.data(1, 1, 1), 1, parallel.subarraytype,
                 0, 22, parallel.comm);
    }

}

// Read the initial temperature distribution from a file
void read_field(Field& field, std::string filename,
                ParallelData& parallel)
{
    std::ifstream file;
    file.open(filename);
    // Read the header
    std::string line, comment;
    std::getline(file, line);
    int nx_full, ny_full;
    std::stringstream(line) >> comment >> nx_full >> ny_full;

    field.setup(nx_full, ny_full, ny_full, parallel);

    // Read the full array
    auto full = Matrix<double> (nx_full, ny_full, ny_full);

    if (0 == parallel.rank) {
        for (int i = 0; i < nx_full; i++)
            for (int j = 0; j < ny_full; j++)
                file >> full(i, j, 0);
    }

    file.close();

    // Inner region (no boundaries)
    auto inner = Matrix<double> (field.nx, field.ny, field.nz);

    MPI_Scatter(full.data(), field.nx * ny_full, MPI_DOUBLE, inner.data(),
                field.nx * ny_full, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Copy to the array containing also boundaries
    for (int i = 0; i < field.nx; i++)
        for (int j = 0; j < field.ny; j++)
             field(i + 1, j + 1, 0) = inner(i, j, 0);

    // Set the boundary values
    for (int i = 0; i < field.nx + 2; i++) {
        // left boundary
        field(i, 0, 0) = field(i, 1, 0);
        // right boundary
        field(i, field.ny + 1, 0) = field(i, field.ny, 0);
    }
    for (int j = 0; j < field.ny + 2; j++) {
        // top boundary
        field.temperature(0, j, 0) = field(1, j, 0);
        // bottom boundary
        field.temperature(field.nx + 1, j, 0) = field.temperature(field.nx, j, 0);
    }

}
