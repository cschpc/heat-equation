// Main solver routines for heat equation solver

#include <mpi.h>

#include "heat.hpp"
#include <Kokkos_Core.hpp>

// Exchange the boundary values
void exchange(Field& field, const ParallelData parallel)
{

    // Send to up, receive from down
    // auto sbuf = field.temperature(1, 0).data();
    auto sbuf = field.temperature.data();
    //auto rbuf  = field.temperature(field.nx + 1, 0).data();
    auto rbuf  = field.temperature.data();
    MPI_Sendrecv(sbuf, field.ny + 2, MPI_DOUBLE,
                 parallel.nup, 11,
                 rbuf, field.ny + 2, MPI_DOUBLE, 
                 parallel.ndown, 11, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    // Send to down, receive from up
    // sbuf = field.temperature(field.nx, 0).data();
    sbuf = field.temperature.data();
    rbuf = field.temperature.data();
    MPI_Sendrecv(sbuf, field.ny + 2, MPI_DOUBLE, 
                 parallel.ndown, 12,
                 rbuf, field.ny + 2, MPI_DOUBLE,
                 parallel.nup, 12, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

}

// Update the temperature values using five-point stencil */
void evolve(Field& curr, Field& prev, const double a, const double dt)
{

  auto inv_dx2 = 1.0 / (prev.dx * prev.dx);    
  auto inv_dy2 = 1.0 / (prev.dx * prev.dx);    

  using MDPolicyType = Kokkos::MDRangePolicy<Kokkos::Rank<2> >;
  MDPolicyType mdpolicy({1, 1}, {curr.nx + 1, curr.ny + 1});

  Kokkos::View<double**, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged> >
       curr_temp (curr.temperature.data(), curr.nx + 2, curr.ny + 2);
  Kokkos::View<const double**, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged> >
       prev_temp (prev.temperature.data(), prev.nx + 2, prev.ny + 2);
  
  Kokkos::parallel_for("evolve", mdpolicy, 
     KOKKOS_LAMBDA(const int i, const int j) {
        curr_temp(i, j) = prev_temp(i, j) + a * dt * (
                 ( prev_temp(i + 1, j) - 2.0 * prev_temp(i, j) + prev_temp(i - 1, j) ) * inv_dx2 +
	         ( prev_temp(i, j + 1) - 2.0 * prev_temp(i, j) + prev_temp(i, j - 1) ) * inv_dy2
                 );
  });
}
