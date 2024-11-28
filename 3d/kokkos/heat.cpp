#include "heat.hpp"
#include "parallel.hpp"
#include "matrix.hpp"
#include <iostream>
#ifndef NO_MPI
#include <mpi.h>
#endif
#include <Kokkos_Core.hpp>

void Field::setup(int nx_in, int ny_in, int nz_in, ParallelData& parallel) 
{
    nx_full = nx_in;
    ny_full = ny_in;
    nz_full = nz_in;

#ifdef NO_MPI
    nx = nx_full;
    ny = ny_full;
    nz = nz_full;
#else
    nx = nx_full / parallel.dims[0];
    if (nx * parallel.dims[0] != nx_full) {
      std::cout << "Cannot divide grid evenly to processors" << std::endl;
      MPI_Abort(MPI_COMM_WORLD, -2);
    }
    ny = ny_full / parallel.dims[1];
    if (ny * parallel.dims[1] != ny_full) {
      std::cout << "Cannot divide grid evenly to processors" << std::endl;
      MPI_Abort(MPI_COMM_WORLD, -2);
    }

    nz = nz_full / parallel.dims[2];
    if (nz * parallel.dims[2] != nz_full) {
      std::cout << "Cannot divide grid evenly to processors" << std::endl;
      MPI_Abort(MPI_COMM_WORLD, -2);
    }
#endif

    // matrix includes also ghost layers
    temperature = Kokkos::View<double***>("T", nx + 2, ny + 2, nz + 2);

#ifndef NO_MPI
    // Communication buffers / datatypes

    parallel.send_buffers[0][0] = Kokkos::View<double**> ("sendbuf_00", ny + 2, nz + 2);
    parallel.send_buffers[0][1] = Kokkos::View<double**> ("sendbuf_01", ny + 2, nz + 2);
    parallel.send_buffers[1][0] = Kokkos::View<double**> ("sendbuf_10", nx + 2, nz + 2);
    parallel.send_buffers[1][1] = Kokkos::View<double**> ("sendbuf_11", nx + 2, nz + 2);
    parallel.send_buffers[2][0] = Kokkos::View<double**> ("sendbuf_20", nx + 2, ny + 2);
    parallel.send_buffers[2][1] = Kokkos::View<double**> ("sendbuf_21", nx + 2, ny + 2);
    parallel.recv_buffers[0][0] = Kokkos::View<double**> ("recvbuf_00", ny + 2, nz + 2);
    parallel.recv_buffers[0][1] = Kokkos::View<double**> ("recvbuf_01", ny + 2, nz + 2);
    parallel.recv_buffers[1][0] = Kokkos::View<double**> ("recvbuf_10", nx + 2, nz + 2);
    parallel.recv_buffers[1][1] = Kokkos::View<double**> ("recvbuf_11", nx + 2, nz + 2);
    parallel.recv_buffers[2][0] = Kokkos::View<double**> ("recvbuf_20", nx + 2, ny + 2);
    parallel.recv_buffers[2][1] = Kokkos::View<double**> ("recvbuf_21", nx + 2, ny + 2);

#endif

}

void Field::generate(const ParallelData& parallel) {

    // Radius of the source disc 
    double radius = (nx_full + ny_full + nz_full) / 18.0;

    using MDPolicyType3D = Kokkos::MDRangePolicy<Kokkos::Rank<3> >;
    MDPolicyType3D mdpolicy_3d({0, 0, 0}, {nx + 2, ny + 2, nz + 2});

    Kokkos::parallel_for("generate_center", mdpolicy_3d,
      KOKKOS_CLASS_LAMBDA(const int i, const int j, const int k) {
                // Distance of point i, j, k from the origin 
                auto dx = i + parallel.coords[0] * nx - nx_full / 2 + 1;
                auto dy = j + parallel.coords[1] * ny - ny_full / 2 + 1;
                auto dz = k + parallel.coords[2] * nz - nz_full / 2 + 1;
                if (dx * dx + dy * dy + dz * dz < radius * radius) {
                    temperature(i, j, k) = 5.0;
                } else {
                    temperature(i, j, k) = 65.0;
                }
            });

    // Boundary conditions
    using MDPolicyType2D = Kokkos::MDRangePolicy<Kokkos::Rank<2> >;

    MDPolicyType2D mdpolicy_z({0, 0}, {nx + 2, ny + 2});
    if (0 == parallel.coords[2])
      Kokkos::parallel_for("generate_z_boundary", mdpolicy_z,
        KOKKOS_CLASS_LAMBDA(const int i, const int j) {
          temperature(i, j, 0) = 20.0;
      });
    if (parallel.coords[2] == parallel.dims[2] - 1)
      Kokkos::parallel_for("generate_z_boundary", mdpolicy_z,
        KOKKOS_CLASS_LAMBDA(const int i, const int j) {
          temperature(i, j, nz + 1) = 35.0;      
      });

    MDPolicyType2D mdpolicy_y({0, 0}, {nx + 2, nz + 2});
    if (0 == parallel.coords[1])
      Kokkos::parallel_for("generate_y_boundary", mdpolicy_y,
        KOKKOS_CLASS_LAMBDA(const int i, const int k) {
          temperature(i, 0, k) = 35.0;
        });
    if (parallel.coords[1] == parallel.dims[1] - 1)
      Kokkos::parallel_for("generate_y_boundary", mdpolicy_y,
        KOKKOS_CLASS_LAMBDA(const int i, const int k) {
          temperature(i, ny + 1, k) = 20.0;
      });
    

    MDPolicyType2D mdpolicy_x({0, 0}, {ny + 2, nz + 2});
    if (0 == parallel.coords[0])
      Kokkos::parallel_for("generate_x_boundary", mdpolicy_x,
        KOKKOS_CLASS_LAMBDA(const int j, const int k) {
          temperature(0, j, k) = 20.0;
        });
    if (parallel.coords[0] == parallel.dims[0] - 1)
      Kokkos::parallel_for("generate_x_boundary", mdpolicy_x,
        KOKKOS_CLASS_LAMBDA(const int j, const int k) {
          temperature(nx + 1, j, k) = 35.0;
        });

}
