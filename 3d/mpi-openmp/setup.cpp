#include <string>
#include <cstdlib>
#include <iostream>
#include "parallel.hpp"
#include "heat.hpp"
#include "functions.hpp"


void initialize(int argc, char *argv[], Field& current,
                Field& previous, int& nsteps, ParallelData& parallel)
{
    /*
     * Following combinations of command line arguments are possible:
     * No arguments:    use default field dimensions and number of time steps
     * One argument:    read initial field from a given file
     * Two arguments:   initial field from file and number of time steps
     * Three arguments: field dimensions (rows,cols) and number of time steps
     */


    int height = 800;             //!< Field dimensions with default values
    int width = 800;
    int length = 800;

    std::string input_file;        //!< Name of the optional input file

    bool read_file = 0;

    nsteps = 500;

    switch (argc) {
    case 1:
        /* Use default values */
        break;
    case 2:
        /* Read initial field from a file */
        input_file = argv[1];
        read_file = true;
        break;
    case 3:
        /* Read initial field from a file */
        input_file = argv[1];
        read_file = true;

        /* Number of time steps */
        nsteps = std::atoi(argv[2]);
        break;
    case 5:
        /* Field dimensions */
        height = std::atoi(argv[1]);
        width = std::atoi(argv[2]);
        length = std::atoi(argv[3]);
        /* Number of time steps */
        nsteps = std::atoi(argv[4]);
        break;
    default:
        std::cout << "Unsupported number of command line arguments" << std::endl;
        exit(-1);
    }

    if (read_file) {
        if (0 == parallel.rank)
            std::cout << "Reading input from " + input_file << std::endl;
        read_field(current, input_file, parallel);
    } else {
        current.setup(height, width, length, parallel);
        current.generate(parallel);
    }

    // copy "current" field also to "previous"
    // previous = current;
    // "manual" assignment in order to ensure first touch
    previous.setup(height, width, length, parallel);
#ifndef NO_FIRST_TOUCH
#pragma omp parallel for collapse(2) schedule(static)
#endif
    for (int i = 0; i < current.nx + 2; i++) {
        for (int j = 0; j < current.ny + 2; j++) {
            for (int k = 0; k < current.nz + 2; k++) {
               previous(i, j, k) = current(i, j, k);
            }
        }
    }

    if (0 == parallel.rank) {
        std::cout << "Simulation parameters: " 
                  << "height: " << height << " width: " << width << " length: " << length
                  << " time steps: " << nsteps << std::endl;
        std::cout << "Number of MPI tasks: " << parallel.size 
                  << " (" << parallel.dims[0] << " x " << parallel.dims[1] << " x " 
                  << parallel.dims[2] << ")" << std::endl;
        std::cout << "Number of OpenMP threads: " << parallel.num_threads << std::endl;
    }
}
