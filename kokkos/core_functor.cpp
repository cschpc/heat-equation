// Main solver routines for heat equation solver

#include <mpi.h>

#include "heat.hpp"
#include <Kokkos_Core.hpp>

// Exchange the boundary values
void exchange(Field& field, const ParallelData parallel)
{

    // Send to up, receive from down
    auto sbuf = Kokkos::subview (field.temperature, 1, Kokkos::ALL).data();
    auto rbuf = Kokkos::subview (field.temperature, field.nx + 1, Kokkos::ALL).data();
    MPI_Sendrecv(sbuf, field.ny + 2, MPI_DOUBLE,
                 parallel.nup, 11,
                 rbuf, field.ny + 2, MPI_DOUBLE, 
                 parallel.ndown, 11, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    // Send to down, receive from up
    sbuf = Kokkos::subview (field.temperature, 
                                    field.nx,
                                    Kokkos::ALL).data();
    rbuf = field.temperature.data();
    MPI_Sendrecv(sbuf, field.ny + 2, MPI_DOUBLE, 
                 parallel.ndown, 12,
                 rbuf, field.ny + 2, MPI_DOUBLE,
                 parallel.nup, 12, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
}

struct evolveFunctor {
  Kokkos::View<double**> curr; 
  Kokkos::View<double**> prev; 
  const double a;
  const double dt;
  double inv_dx2, inv_dy2;

  evolveFunctor(Field& curr_, Field& prev_, const double a_, const double dt_) :
     curr(curr_.temperature), prev(prev_.temperature), a(a_), dt(dt_) {
    inv_dx2 = 1.0 / (prev_.dx * prev_.dx);  
    inv_dy2 = 1.0 / (prev_.dy * prev_.dy);  
  }

  KOKKOS_INLINE_FUNCTION
  void operator()(const int i, const int j) const {
        curr(i, j) = prev(i, j) + a * dt * (
                 ( prev(i + 1, j) - 2.0 * prev(i, j) + prev(i - 1, j) ) * inv_dx2 +
	         ( prev(i, j + 1) - 2.0 * prev(i, j) + prev(i, j - 1) ) * inv_dy2
                 );
  };
};

// Update the temperature values using five-point stencil */
void evolve(Field& curr, Field& prev, const double a, const double dt)
{

  using MDPolicyType = Kokkos::MDRangePolicy<Kokkos::Rank<2> >;
  MDPolicyType mdpolicy({1, 1}, {curr.nx + 1, curr.ny + 1});

  Kokkos::parallel_for("evolve", mdpolicy, evolveFunctor(curr, prev, a, dt));
}
