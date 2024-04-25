#include "heat.hpp"
#include "input.hpp"
#include <iostream>
#include <string>

void initialize(const heat::Input &input, Field &current, Field &previous,
                const ParallelData &parallel) {
    if (input.read_file) {
        if (0 == parallel.rank)
            std::cout << "Reading input from " + input.fname << std::endl;
        read_field(current, input.fname, parallel);
    } else {
        current.setup(input.rows, input.cols, parallel);
        current.generate(parallel);
    }

    // copy "current" field also to "previous"
    previous = current;

    if (0 == parallel.rank) {
        std::cout << "Simulation parameters: "
                  << "rows: " << input.rows << " columns: " << input.cols
                  << " time steps: " << input.nsteps << std::endl;
        std::cout << "Number of MPI tasks: " << parallel.size << std::endl;
    }
}
