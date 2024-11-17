#include "heat.hpp"
#include "matrix.hpp"
#include <iostream>
#include <mpi.h>
#include <Kokkos_Core.hpp>

void Field::setup(int nx_in, int ny_in, ParallelData parallel) 
{
    nx_full = nx_in;
    ny_full = ny_in;

    nx = nx_full / parallel.size;
    if (nx * parallel.size != nx_full) {
        std::cout << "Cannot divide grid evenly to processors" << std::endl;
        MPI_Abort(MPI_COMM_WORLD, -2);
    }
    ny = ny_full;

   // matrix includes also ghost layers
   temperature = Kokkos::View<double**>("T", nx + 2, ny + 2);
}

void Field::generate(ParallelData parallel) {

    // Radius of the source disc 
    auto radius = nx_full / 6.0;

    //Kokkos::View<double**, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged> >
    //   temperature_view (temperature.data(), nx + 2, ny + 2);

    using MDPolicyType2D = Kokkos::MDRangePolicy<Kokkos::Rank<2> >;
    MDPolicyType2D mdpolicy_2d({0, 0}, {nx + 2, ny + 2}); 

    Kokkos::parallel_for("generate_center", mdpolicy_2d, 
      KOKKOS_LAMBDA(const int i, const int j) {
            // Distance of point i, j from the origin 
            auto dx = i + parallel.rank * nx - nx_full / 2 + 1;
            auto dy = j - ny / 2 + 1;
            if (dx * dx + dy * dy < radius * radius) {
                temperature(i, j) = 5.0;
            } else {
                temperature(i, j) = 65.0;
            }
        });

    // Boundary conditions
    Kokkos::parallel_for("generate_x_boundary", nx + 2,
      KOKKOS_LAMBDA(const int i) {
        // Left
        temperature(i, 0) = 20.0;
        // Right
        temperature(i, ny + 1) = 70.0;
      });

    // Top
    if (0 == parallel.rank) {
      Kokkos::parallel_for("generate_x_boundary", ny + 2,
        KOKKOS_LAMBDA(const int j) {
            temperature(0, j) = 85.0;
        });
    }
    // Bottom
    if (parallel.rank == parallel.size - 1) {
      Kokkos::parallel_for("generate_x_boundary", ny + 2,
        KOKKOS_LAMBDA(const int j) {
            temperature(nx + 1, j) = 5.0;
        });
    }
}
