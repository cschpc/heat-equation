// Utility functions for heat equation solver

#include <algorithm>
#include <mpi.h>
#include <tuple>

#include "field.hpp"
#include "parallel.hpp"
#include "utilities.hpp"

namespace heat {
// Calculate average temperature
double average(const Field &field, const ParallelData &parallel) {
    double average = 0.0;
    double local_sum = field.sum();

    MPI_Allreduce(&local_sum, &average, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    average /= (field.num_rows * field.num_cols * parallel.size);
    return average;
}

std::tuple<int, int, std::vector<double>> generate_field(int num_rows,
                                                         int num_cols) {
    std::vector<double> data(num_rows * num_cols);
    // Radius of the source disc
    const auto radius = num_rows / 6.0;
    for (int i = 0; i < num_rows; i++) {
        for (int j = 0; j < num_cols; j++) {
            const auto index = i * num_cols + j;
            // Distance of point i, j from the origin
            const auto dx = i - num_rows / 2 + 2;
            const auto dy = j - num_cols / 2 + 2;
            data[index] = dx * dx + dy * dy < radius * radius ? 5.0 : 65.0;
        }
    }
    return std::make_tuple(num_rows, num_cols, data);
}

std::vector<double> scatter(std::vector<double> &&full_data,
                            int num_values_per_rank, int n) {
    std::vector<double> my_data(num_values_per_rank);
    MPI_Scatter(full_data.data(), num_values_per_rank, MPI_DOUBLE,
                my_data.data(), num_values_per_rank, MPI_DOUBLE, 0,
                MPI_COMM_WORLD);

    return my_data;
}

std::vector<double> gather(const Field &field, const ParallelData &parallel) {
    std::vector<double> full_data;
    constexpr auto tag = 22;
    const auto num_values = field.num_rows * parallel.size * field.num_cols;
    auto data = field.get_data();

    if (0 == parallel.rank) {
        full_data.reserve(num_values);
        std::copy_n(data.begin(), data.size(), full_data.end());

        // Receive data from other ranks
        for (int from = 1; from < parallel.size; from++) {
            MPI_Recv(data.data(), data.size(), MPI_DOUBLE, from, tag,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            std::copy_n(data.begin(), data.size(), full_data.end());
        }
    } else {
        constexpr int to = 0;
        MPI_Send(data.data(), data.size(), MPI_DOUBLE, to, tag, MPI_COMM_WORLD);
    }

    return full_data;
}
} // namespace heat
