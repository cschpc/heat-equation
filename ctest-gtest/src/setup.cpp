#include "field.hpp"
#include "input.hpp"
#include "io.hpp"
#include "parallel.hpp"
#include <iostream>

// TODO:
// - use heat::read_field or heat::generate_field to get data
// - partition the data to the processes/ranks in a similar fashion as is done
// in io.cpp:read_field, i.e. use MPI_scatter with the full_data
// - add constructor to field that takes (local) num_rows and num_cols and local
// data
// - in the constructor generate the ghost data as in io.cpp:read_field
// - while doing this, add tests for each section (before if possible)
Field initialize(const heat::Input &input, const ParallelData &parallel) {
    Field field = {};
    if (input.read_file) {
        if (0 == parallel.rank) {
            std::cout << "Reading input from " + input.fname << std::endl;
        }
        read_field(field, input.fname, parallel);
    } else {
        field.setup(input.rows, input.cols, parallel);
        field.generate(parallel);
    }

    if (0 == parallel.rank) {
        std::cout << "Simulation parameters: "
                  << "rows: " << input.rows << " columns: " << input.cols
                  << " time steps: " << input.nsteps << std::endl;
        std::cout << "Number of MPI tasks: " << parallel.size << std::endl;
    }
    return field;
}
