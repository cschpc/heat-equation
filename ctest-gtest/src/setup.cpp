#include "field.hpp"
#include "io.hpp"
#include "parallel.hpp"
#include "utilities.hpp"
#include <iostream>

Field initialize(const heat::Input &input, const ParallelData &parallel) {
    int num_rows_global = 0;
    int num_cols_global = 0;
    std::vector<double> data;

    if (input.read_file) {
        if (0 == parallel.rank) {
            std::cout << "Reading input from " + input.fname << std::endl;
        }
        auto tuple = heat::read_field(input.fname, parallel.rank);
        num_rows_global = std::get<0>(tuple);
        num_cols_global = std::get<1>(tuple);
        data = std::get<2>(tuple);
    } else {
        if (0 == parallel.rank) {
            std::cout << "Generating data" << std::endl;
        }
        num_rows_global = input.rows;
        num_cols_global = input.cols;
        data = heat::generate_field(num_rows_global, num_cols_global,
                                    parallel.rank);
    }

    if (0 == parallel.rank) {
        std::cout << "Simulation parameters: "
                  << "rows: " << num_rows_global
                  << " columns: " << num_cols_global
                  << " time steps: " << input.nsteps << std::endl;
        std::cout << "Number of MPI tasks: " << parallel.size << std::endl;
    }

    auto [num_rows, num_cols] = Field::partition_domain(
        num_rows_global, num_cols_global, parallel.size);

    return Field(
        heat::scatter(std::move(data), num_rows * num_cols, parallel.size),
        num_rows, num_cols);
}
