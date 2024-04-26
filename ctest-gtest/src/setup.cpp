#include "field.hpp"
#include "heat.hpp"
#include "input.hpp"
#include "parallel.hpp"
#include <iostream>

Field initialize(const heat::Input &input, const ParallelData &parallel) {
    Field field = {};
    if (input.read_file) {
        if (0 == parallel.rank)
            std::cout << "Reading input from " + input.fname << std::endl;
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
