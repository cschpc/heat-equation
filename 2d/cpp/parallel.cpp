// SPDX-FileCopyrightText: 2021 CSC - IT Center for Science Ltd. <www.csc.fi>
//
// SPDX-License-Identifier: MIT

#include "heat.hpp"
#include <mpi.h>

ParallelData::ParallelData() {

    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    nup = rank - 1;
    ndown = rank + 1;

    if (nup < 0) {
        nup = MPI_PROC_NULL;
    }
    if (ndown > size - 1) {
        ndown = MPI_PROC_NULL;
    }
}
